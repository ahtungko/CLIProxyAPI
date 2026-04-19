package executor

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	"github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/usage"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"github.com/tiktoken-go/tokenizer"
)

const (
	freebuffAPIBase             = "https://www.codebuff.com"
	freebuffUserAgent           = "freebuff-proxy/1.0"
	freebuffSessionPollInterval = 5 * time.Second
	freebuffSessionWaitTimeout  = 15 * time.Minute
)

// freebuffModelToAgent maps model IDs to Freebuff agent identifiers.
var freebuffModelToAgent = map[string]string{
	"minimax/minimax-m2.7":                 "base2-free", // legacy alias
	"z-ai/glm-5.1":                         "base2-free",
	"google/gemini-2.5-flash-lite":         "file-picker",
	"google/gemini-3.1-flash-lite-preview": "file-picker-max",
}

// freebuffRunCache stores the cached Agent Run IDs per auth token + agent pair.
type freebuffRunCache struct {
	mu   sync.RWMutex
	runs map[string]string // key: "{tokenSuffix}:{agentID}" -> runID
}

var globalFreebuffRunCache = &freebuffRunCache{
	runs: make(map[string]string),
}

type freebuffSessionState struct {
	mu         sync.Mutex
	instanceID string
	disabled   bool
}

type freebuffSessionCache struct {
	mu       sync.Mutex
	sessions map[string]*freebuffSessionState
}

var globalFreebuffSessionCache = &freebuffSessionCache{
	sessions: make(map[string]*freebuffSessionState),
}

type freebuffSessionResponse struct {
	Status                 string `json:"status"`
	InstanceID             string `json:"instanceId"`
	Position               int    `json:"position"`
	QueueDepth             int    `json:"queueDepth"`
	EstimatedWaitMs        int64  `json:"estimatedWaitMs"`
	RemainingMs            int64  `json:"remainingMs"`
	GracePeriodRemainingMs int64  `json:"gracePeriodRemainingMs"`
	Message                string `json:"message"`
}

func (c *freebuffRunCache) get(key string) (string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	v, ok := c.runs[key]
	return v, ok
}

func (c *freebuffRunCache) set(key, runID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.runs[key] = runID
}

func (c *freebuffRunCache) invalidate(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.runs, key)
}

func (c *freebuffSessionCache) state(key string) *freebuffSessionState {
	c.mu.Lock()
	defer c.mu.Unlock()
	if state, ok := c.sessions[key]; ok {
		return state
	}
	state := &freebuffSessionState{}
	c.sessions[key] = state
	return state
}

func (c *freebuffSessionCache) invalidate(key string) {
	state := c.state(key)
	state.mu.Lock()
	defer state.mu.Unlock()
	state.instanceID = ""
	state.disabled = false
}

// FreebuffExecutor implements a stateless executor for the Freebuff (Codebuff) proxy.
// It manages Agent Run lifecycle, injects codebuff_metadata, and proxies OpenAI-compatible
// chat completion requests to the Freebuff upstream API.
type FreebuffExecutor struct {
	cfg *config.Config
}

// NewFreebuffExecutor creates a new FreebuffExecutor.
func NewFreebuffExecutor(cfg *config.Config) *FreebuffExecutor {
	return &FreebuffExecutor{cfg: cfg}
}

// Identifier implements cliproxyauth.ProviderExecutor.
func (e *FreebuffExecutor) Identifier() string { return "freebuff" }

// PrepareRequest injects Freebuff credentials into the outgoing HTTP request.
func (e *FreebuffExecutor) PrepareRequest(req *http.Request, auth *cliproxyauth.Auth) error {
	if req == nil {
		return nil
	}
	_, authToken := e.resolveCredentials(auth)
	if strings.TrimSpace(authToken) != "" {
		req.Header.Set("Authorization", "Bearer "+authToken)
	}
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(req, attrs)
	return nil
}

// HttpRequest injects Freebuff credentials into the request and executes it.
func (e *FreebuffExecutor) HttpRequest(ctx context.Context, auth *cliproxyauth.Auth, req *http.Request) (*http.Response, error) {
	if req == nil {
		return nil, fmt.Errorf("freebuff executor: request is nil")
	}
	if ctx == nil {
		ctx = req.Context()
	}
	httpReq := req.WithContext(ctx)
	if err := e.PrepareRequest(httpReq, auth); err != nil {
		return nil, err
	}
	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	return httpClient.Do(httpReq)
}

// Execute handles non-streaming requests to the Freebuff API.
func (e *FreebuffExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	reporter := helps.NewUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	baseURL, authToken := e.resolveCredentials(auth)
	if authToken == "" {
		err = statusErr{code: http.StatusUnauthorized, msg: "freebuff executor: missing auth token"}
		return
	}
	if baseURL == "" {
		baseURL = freebuffAPIBase
	}

	from := opts.SourceFormat
	to := sdktranslator.FromString("openai")
	originalPayloadSource := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayloadSource = opts.OriginalRequest
	}
	originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayloadSource, opts.Stream)
	translated := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, opts.Stream)
	requestedModel := helps.PayloadRequestedModel(opts, req.Model)
	translated = helps.ApplyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", translated, originalTranslated, requestedModel)

	translated, err = thinking.ApplyThinking(translated, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return resp, err
	}

	// Ensure the upstream sees the base model name.
	translated, _ = sjson.SetBytes(translated, "model", baseModel)
	// Force streaming for upstream; Freebuff always streams.
	translated, _ = sjson.SetBytes(translated, "stream", true)

	// Get or create Agent Run.
	agentID := freebuffAgentForModel(baseModel)
	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	runID, err := e.getOrCreateRun(ctx, httpClient, baseURL, authToken, agentID)
	if err != nil {
		return resp, err
	}

	instanceID, err := e.ensureActiveSession(ctx, httpClient, baseURL, authToken)
	if err != nil {
		return resp, err
	}

	// Inject codebuff_metadata.
	translated = injectFreebuffMetadata(translated, runID, instanceID)

	url := strings.TrimSuffix(baseURL, "/") + "/api/v1/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(translated))
	if err != nil {
		return resp, err
	}
	applyFreebuffHeaders(httpReq, authToken, true)
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      translated,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())

	// Handle expired Agent Run: retry once on 400/404.
	if httpResp.StatusCode == 400 || httpResp.StatusCode == 404 {
		b, _ := io.ReadAll(httpResp.Body)
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("freebuff executor: close response body error: %v", errClose)
		}
		helps.AppendAPIResponseChunk(ctx, e.cfg, b)
		log.Debugf("freebuff executor: run may be stale (HTTP %d), recreating", httpResp.StatusCode)
		cacheKey := freebuffRunCacheKey(authToken, agentID)
		globalFreebuffRunCache.invalidate(cacheKey)

		runID, err = e.createRun(ctx, httpClient, baseURL, authToken, agentID)
		if err != nil {
			return resp, err
		}
		globalFreebuffRunCache.set(cacheKey, runID)
		translated = injectFreebuffMetadata(translated, runID, instanceID)

		retryReq, errRetry := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(translated))
		if errRetry != nil {
			return resp, errRetry
		}
		applyFreebuffHeaders(retryReq, authToken, true)
		httpResp, err = httpClient.Do(retryReq)
		if err != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, err)
			return resp, err
		}
		helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	}
	if retryBody, shouldRetrySession := shouldRetryFreebuffSession(httpResp); shouldRetrySession {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("freebuff executor: close response body error: %v", errClose)
		}
		helps.AppendAPIResponseChunk(ctx, e.cfg, retryBody)
		globalFreebuffSessionCache.invalidate(freebuffSessionCacheKey(authToken))

		instanceID, err = e.ensureActiveSession(ctx, httpClient, baseURL, authToken)
		if err != nil {
			return resp, err
		}
		translated = injectFreebuffMetadata(translated, runID, instanceID)

		retryReq, errRetry := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(translated))
		if errRetry != nil {
			return resp, errRetry
		}
		applyFreebuffHeaders(retryReq, authToken, true)
		httpResp, err = httpClient.Do(retryReq)
		if err != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, err)
			return resp, err
		}
		helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("freebuff executor: close response body error: %v", errClose)
		}
	}()

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		helps.AppendAPIResponseChunk(ctx, e.cfg, b)
		helps.LogWithRequestID(ctx).Debugf("freebuff request error, status: %d, message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		err = statusErr{code: httpResp.StatusCode, msg: string(b)}
		return resp, err
	}

	// Collect SSE stream into a single non-streaming response.
	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	helps.AppendAPIResponseChunk(ctx, e.cfg, body)

	assembledResponse := assembleFreebuffSSEResponse(body, baseModel)
	reporter.Publish(ctx, freebuffUsageDetailOrEstimate(baseModel, translated, assembledResponse))
	reporter.EnsurePublished(ctx)

	var param any
	out := sdktranslator.TranslateNonStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, assembledResponse, &param)
	resp = cliproxyexecutor.Response{Payload: out, Headers: httpResp.Header.Clone()}
	return resp, nil
}

// ExecuteStream handles streaming requests to the Freebuff API.
func (e *FreebuffExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	reporter := helps.NewUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	baseURL, authToken := e.resolveCredentials(auth)
	if authToken == "" {
		err = statusErr{code: http.StatusUnauthorized, msg: "freebuff executor: missing auth token"}
		return nil, err
	}
	if baseURL == "" {
		baseURL = freebuffAPIBase
	}

	from := opts.SourceFormat
	to := sdktranslator.FromString("openai")
	originalPayloadSource := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayloadSource = opts.OriginalRequest
	}
	originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayloadSource, true)
	translated := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, true)
	requestedModel := helps.PayloadRequestedModel(opts, req.Model)
	translated = helps.ApplyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", translated, originalTranslated, requestedModel)

	translated, err = thinking.ApplyThinking(translated, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return nil, err
	}

	translated, _ = sjson.SetBytes(translated, "model", baseModel)
	translated, _ = sjson.SetBytes(translated, "stream", true)
	translated, _ = sjson.SetBytes(translated, "stream_options.include_usage", true)

	agentID := freebuffAgentForModel(baseModel)
	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	runID, err := e.getOrCreateRun(ctx, httpClient, baseURL, authToken, agentID)
	if err != nil {
		return nil, err
	}

	instanceID, err := e.ensureActiveSession(ctx, httpClient, baseURL, authToken)
	if err != nil {
		return nil, err
	}

	translated = injectFreebuffMetadata(translated, runID, instanceID)

	url := strings.TrimSuffix(baseURL, "/") + "/api/v1/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(translated))
	if err != nil {
		return nil, err
	}
	applyFreebuffHeaders(httpReq, authToken, true)
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      translated,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, err)
		return nil, err
	}
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())

	// Handle expired Agent Run: retry once on 400/404.
	if httpResp.StatusCode == 400 || httpResp.StatusCode == 404 {
		b, _ := io.ReadAll(httpResp.Body)
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("freebuff executor: close response body error: %v", errClose)
		}
		helps.AppendAPIResponseChunk(ctx, e.cfg, b)
		log.Debugf("freebuff executor: run may be stale (HTTP %d), recreating", httpResp.StatusCode)
		cacheKey := freebuffRunCacheKey(authToken, agentID)
		globalFreebuffRunCache.invalidate(cacheKey)

		runID, err = e.createRun(ctx, httpClient, baseURL, authToken, agentID)
		if err != nil {
			return nil, err
		}
		globalFreebuffRunCache.set(cacheKey, runID)
		translated = injectFreebuffMetadata(translated, runID, instanceID)

		retryReq, errRetry := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(translated))
		if errRetry != nil {
			return nil, errRetry
		}
		applyFreebuffHeaders(retryReq, authToken, true)
		httpResp, err = httpClient.Do(retryReq)
		if err != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, err)
			return nil, err
		}
		helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	}
	if retryBody, shouldRetrySession := shouldRetryFreebuffSession(httpResp); shouldRetrySession {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("freebuff executor: close response body error: %v", errClose)
		}
		helps.AppendAPIResponseChunk(ctx, e.cfg, retryBody)
		globalFreebuffSessionCache.invalidate(freebuffSessionCacheKey(authToken))

		instanceID, err = e.ensureActiveSession(ctx, httpClient, baseURL, authToken)
		if err != nil {
			return nil, err
		}
		translated = injectFreebuffMetadata(translated, runID, instanceID)

		retryReq, errRetry := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(translated))
		if errRetry != nil {
			return nil, errRetry
		}
		applyFreebuffHeaders(retryReq, authToken, true)
		httpResp, err = httpClient.Do(retryReq)
		if err != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, err)
			return nil, err
		}
		helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		data, readErr := io.ReadAll(httpResp.Body)
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("freebuff executor: close response body error: %v", errClose)
		}
		if readErr != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, readErr)
			return nil, readErr
		}
		helps.AppendAPIResponseChunk(ctx, e.cfg, data)
		helps.LogWithRequestID(ctx).Debugf("freebuff request error, status: %d, message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), data))
		err = statusErr{code: httpResp.StatusCode, msg: string(data)}
		return nil, err
	}

	out := make(chan cliproxyexecutor.StreamChunk)
	estimatedInputTokens := freebuffEstimateInputTokens(baseModel, translated)
	go func() {
		defer close(out)
		defer func() {
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("freebuff executor: close response body error: %v", errClose)
			}
		}()
		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(nil, 52_428_800) // 50MB
		var param any
		var sawUsage bool
		outputSegments := make([]string, 0, 32)
		for scanner.Scan() {
			line := scanner.Bytes()
			helps.AppendAPIResponseChunk(ctx, e.cfg, line)
			if detail, ok := helps.ParseOpenAIStreamUsage(line); ok {
				sawUsage = true
				reporter.Publish(ctx, detail)
			}
			if len(line) == 0 {
				continue
			}
			if !bytes.HasPrefix(line, []byte("data:")) {
				continue
			}
			freebuffCollectOutputSegments(line, &outputSegments)

			chunks := sdktranslator.TranslateStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, bytes.Clone(line), &param)
			for i := range chunks {
				out <- cliproxyexecutor.StreamChunk{Payload: chunks[i]}
			}
		}
		if errScan := scanner.Err(); errScan != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, errScan)
			reporter.PublishFailure(ctx)
			out <- cliproxyexecutor.StreamChunk{Err: errScan}
		} else {
			// Feed synthetic done marker to ensure translator emits final events.
			chunks := sdktranslator.TranslateStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, []byte("data: [DONE]"), &param)
			for i := range chunks {
				out <- cliproxyexecutor.StreamChunk{Payload: chunks[i]}
			}
			if !sawUsage {
				reporter.Publish(ctx, freebuffEstimateUsageFromSegments(baseModel, estimatedInputTokens, outputSegments))
			}
		}
		reporter.EnsurePublished(ctx)
	}()
	return &cliproxyexecutor.StreamResult{Headers: httpResp.Header.Clone(), Chunks: out}, nil
}

// CountTokens provides approximate token counting for Freebuff models.
func (e *FreebuffExecutor) CountTokens(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	from := opts.SourceFormat
	to := sdktranslator.FromString("openai")
	translated := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, false)

	translated, err := thinking.ApplyThinking(translated, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}

	enc, err := helps.TokenizerForModel(baseModel)
	if err != nil {
		return cliproxyexecutor.Response{}, fmt.Errorf("freebuff executor: tokenizer init failed: %w", err)
	}

	count, err := helps.CountOpenAIChatTokens(enc, translated)
	if err != nil {
		return cliproxyexecutor.Response{}, fmt.Errorf("freebuff executor: token counting failed: %w", err)
	}

	usageJSON := helps.BuildOpenAIUsageJSON(count)
	translatedUsage := sdktranslator.TranslateTokenCount(ctx, to, from, count, usageJSON)
	return cliproxyexecutor.Response{Payload: translatedUsage}, nil
}

// Refresh is a no-op for Freebuff since auth tokens don't auto-expire in the OAuth sense.
func (e *FreebuffExecutor) Refresh(ctx context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	log.Debugf("freebuff executor: refresh called")
	_ = ctx
	return auth, nil
}

// resolveCredentials extracts the base URL and auth token from the auth object.
func (e *FreebuffExecutor) resolveCredentials(auth *cliproxyauth.Auth) (baseURL, authToken string) {
	if auth == nil {
		return "", ""
	}
	if auth.Attributes != nil {
		baseURL = strings.TrimSpace(auth.Attributes["base_url"])
		authToken = strings.TrimSpace(auth.Attributes["api_key"])
	}
	if authToken == "" && auth.Metadata != nil {
		if v, ok := auth.Metadata["authToken"].(string); ok {
			authToken = strings.TrimSpace(v)
		}
	}
	return
}

// getOrCreateRun returns a cached run ID or creates a new Agent Run.
func (e *FreebuffExecutor) getOrCreateRun(ctx context.Context, client *http.Client, baseURL, authToken, agentID string) (string, error) {
	key := freebuffRunCacheKey(authToken, agentID)
	if runID, ok := globalFreebuffRunCache.get(key); ok && runID != "" {
		return runID, nil
	}
	runID, err := e.createRun(ctx, client, baseURL, authToken, agentID)
	if err != nil {
		return "", err
	}
	globalFreebuffRunCache.set(key, runID)
	return runID, nil
}

func (e *FreebuffExecutor) ensureActiveSession(ctx context.Context, client *http.Client, baseURL, authToken string) (string, error) {
	state := globalFreebuffSessionCache.state(freebuffSessionCacheKey(authToken))
	state.mu.Lock()
	defer state.mu.Unlock()

	if state.disabled {
		return "", nil
	}
	if strings.TrimSpace(state.instanceID) != "" {
		return state.instanceID, nil
	}

	for attempt := 0; attempt < 2; attempt++ {
		sessionResp, err := e.postSession(ctx, client, baseURL, authToken)
		if err != nil {
			return "", err
		}
		instanceID, retry, err := e.resolveSessionStateLocked(ctx, client, baseURL, authToken, state, sessionResp)
		if err != nil {
			return "", err
		}
		if !retry {
			return instanceID, nil
		}
	}

	return "", statusErr{code: http.StatusConflict, msg: "freebuff executor: failed to establish active session"}
}

func (e *FreebuffExecutor) resolveSessionStateLocked(ctx context.Context, client *http.Client, baseURL, authToken string, state *freebuffSessionState, sessionResp freebuffSessionResponse) (string, bool, error) {
	switch sessionResp.Status {
	case "disabled":
		state.disabled = true
		state.instanceID = ""
		return "", false, nil
	case "active":
		if strings.TrimSpace(sessionResp.InstanceID) == "" {
			return "", false, statusErr{code: http.StatusBadGateway, msg: "freebuff executor: active session missing instance id"}
		}
		state.disabled = false
		state.instanceID = sessionResp.InstanceID
		return state.instanceID, false, nil
	case "ended":
		if strings.TrimSpace(sessionResp.InstanceID) == "" {
			state.instanceID = ""
			return "", true, nil
		}
		state.disabled = false
		state.instanceID = sessionResp.InstanceID
		return state.instanceID, false, nil
	case "queued":
		if strings.TrimSpace(sessionResp.InstanceID) == "" {
			return "", false, statusErr{code: http.StatusBadGateway, msg: "freebuff executor: queued session missing instance id"}
		}
		state.disabled = false
		state.instanceID = sessionResp.InstanceID
		return e.pollUntilActiveSessionLocked(ctx, client, baseURL, authToken, state, sessionResp)
	case "none", "superseded":
		state.instanceID = ""
		state.disabled = false
		return "", true, nil
	default:
		raw, _ := json.Marshal(sessionResp)
		return "", false, statusErr{
			code: http.StatusBadGateway,
			msg:  fmt.Sprintf("freebuff executor: unexpected session status %q: %s", sessionResp.Status, string(raw)),
		}
	}
}

func (e *FreebuffExecutor) pollUntilActiveSessionLocked(ctx context.Context, client *http.Client, baseURL, authToken string, state *freebuffSessionState, sessionResp freebuffSessionResponse) (string, bool, error) {
	waitBudget := freebuffSessionWaitTimeout
	if sessionResp.EstimatedWaitMs > 0 {
		estimated := time.Duration(sessionResp.EstimatedWaitMs)*time.Millisecond + 30*time.Second
		if estimated > waitBudget {
			waitBudget = estimated
		}
	}
	if waitBudget < 30*time.Second {
		waitBudget = 30 * time.Second
	}

	waitCtx := ctx
	var cancel context.CancelFunc
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		waitCtx, cancel = context.WithTimeout(ctx, waitBudget)
		defer cancel()
	}

	ticker := time.NewTicker(freebuffSessionPollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-waitCtx.Done():
			return "", false, statusErr{code: http.StatusTooManyRequests, msg: "freebuff executor: timed out waiting for freebuff session admission"}
		case <-ticker.C:
		}

		polled, err := e.getSession(waitCtx, client, baseURL, authToken, state.instanceID)
		if err != nil {
			return "", false, err
		}
		switch polled.Status {
		case "disabled":
			state.disabled = true
			state.instanceID = ""
			return "", false, nil
		case "active":
			if strings.TrimSpace(polled.InstanceID) == "" {
				return "", false, statusErr{code: http.StatusBadGateway, msg: "freebuff executor: active session missing instance id"}
			}
			state.instanceID = polled.InstanceID
			return state.instanceID, false, nil
		case "ended":
			if strings.TrimSpace(polled.InstanceID) == "" {
				state.instanceID = ""
				return "", true, nil
			}
			state.instanceID = polled.InstanceID
			return state.instanceID, false, nil
		case "queued":
			if strings.TrimSpace(polled.InstanceID) != "" {
				state.instanceID = polled.InstanceID
			}
		case "none", "superseded":
			state.instanceID = ""
			return "", true, nil
		default:
			raw, _ := json.Marshal(polled)
			return "", false, statusErr{
				code: http.StatusBadGateway,
				msg:  fmt.Sprintf("freebuff executor: unexpected polled session status %q: %s", polled.Status, string(raw)),
			}
		}
	}
}

func (e *FreebuffExecutor) postSession(ctx context.Context, client *http.Client, baseURL, authToken string) (freebuffSessionResponse, error) {
	return e.doSessionRequest(ctx, client, baseURL, authToken, http.MethodPost, "", "")
}

func (e *FreebuffExecutor) getSession(ctx context.Context, client *http.Client, baseURL, authToken, instanceID string) (freebuffSessionResponse, error) {
	return e.doSessionRequest(ctx, client, baseURL, authToken, http.MethodGet, instanceID, "")
}

func (e *FreebuffExecutor) doSessionRequest(ctx context.Context, client *http.Client, baseURL, authToken, method, instanceID, body string) (freebuffSessionResponse, error) {
	var payload io.Reader
	if body != "" {
		payload = strings.NewReader(body)
	}

	url := strings.TrimSuffix(baseURL, "/") + "/api/v1/freebuff/session"
	req, err := http.NewRequestWithContext(ctx, method, url, payload)
	if err != nil {
		return freebuffSessionResponse{}, err
	}
	req.Header.Set("Authorization", "Bearer "+authToken)
	req.Header.Set("User-Agent", freebuffUserAgent)
	req.Header.Set("Accept", "application/json")
	if strings.TrimSpace(instanceID) != "" {
		req.Header.Set("X-Freebuff-Instance-Id", instanceID)
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := client.Do(req)
	if err != nil {
		return freebuffSessionResponse{}, err
	}
	defer func() {
		if errClose := resp.Body.Close(); errClose != nil {
			log.Errorf("freebuff executor: close session response body error: %v", errClose)
		}
	}()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return freebuffSessionResponse{}, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return freebuffSessionResponse{}, statusErr{code: resp.StatusCode, msg: string(respBody)}
	}

	var sessionResp freebuffSessionResponse
	if err := json.Unmarshal(respBody, &sessionResp); err != nil {
		return freebuffSessionResponse{}, fmt.Errorf("freebuff executor: failed to decode session response: %w", err)
	}
	return sessionResp, nil
}

// createRun creates a new Agent Run via the Freebuff API.
// It uses the resolved baseURL so that custom gateways/mirrors are respected.
func (e *FreebuffExecutor) createRun(ctx context.Context, client *http.Client, baseURL, authToken, agentID string) (string, error) {
	body := map[string]string{
		"action":  "START",
		"agentId": agentID,
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("freebuff executor: failed to marshal run request: %w", err)
	}

	reqURL := strings.TrimSuffix(baseURL, "/") + "/api/v1/agent-runs"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, bytes.NewReader(payload))
	if err != nil {
		return "", fmt.Errorf("freebuff executor: failed to create run request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+authToken)
	req.Header.Set("User-Agent", freebuffUserAgent)

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("freebuff executor: agent-runs request failed: %w", err)
	}
	defer func() {
		if errClose := resp.Body.Close(); errClose != nil {
			log.Errorf("freebuff executor: close agent-runs response body error: %v", errClose)
		}
	}()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("freebuff executor: failed to read agent-runs response: %w", err)
	}
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("freebuff executor: agent-runs failed (HTTP %d): %s", resp.StatusCode, string(respBody))
	}

	runID := gjson.GetBytes(respBody, "runId").String()
	if runID == "" {
		return "", fmt.Errorf("freebuff executor: agent-runs returned empty runId")
	}

	log.Debugf("freebuff executor: created agent run %s for agent %s", runID, agentID)
	return runID, nil
}

// freebuffAgentForModel maps a model name to the corresponding Freebuff agent ID.
func freebuffAgentForModel(model string) string {
	if agentID, ok := freebuffModelToAgent[model]; ok {
		return agentID
	}
	return "base2-free"
}

// freebuffRunCacheKey builds a cache key for the run cache.
func freebuffRunCacheKey(authToken, agentID string) string {
	// Use last 8 chars of token to avoid storing full token in memory as key.
	tokenSuffix := authToken
	if len(tokenSuffix) > 8 {
		tokenSuffix = tokenSuffix[len(tokenSuffix)-8:]
	}
	return tokenSuffix + ":" + agentID
}

func freebuffSessionCacheKey(authToken string) string {
	return freebuffRunCacheKey(authToken, "session")
}

// injectFreebuffMetadata injects the codebuff_metadata into the payload.
func injectFreebuffMetadata(payload []byte, runID string, instanceID string) []byte {
	clientID := fmt.Sprintf("freebuff-proxy-%s", freebuffRandomAlphanumeric(8))
	payload, _ = sjson.SetBytes(payload, "codebuff_metadata.run_id", runID)
	payload, _ = sjson.SetBytes(payload, "codebuff_metadata.client_id", clientID)
	payload, _ = sjson.SetBytes(payload, "codebuff_metadata.cost_mode", "free")
	if strings.TrimSpace(instanceID) != "" {
		payload, _ = sjson.SetBytes(payload, "codebuff_metadata.freebuff_instance_id", instanceID)
	}
	return payload
}

func shouldRetryFreebuffSession(resp *http.Response) ([]byte, bool) {
	if resp == nil {
		return nil, false
	}
	switch resp.StatusCode {
	case http.StatusConflict, http.StatusGone, http.StatusPreconditionRequired, http.StatusTooManyRequests, http.StatusUpgradeRequired:
	default:
		return nil, false
	}
	body, _ := io.ReadAll(resp.Body)
	errCode := gjson.GetBytes(body, "error").String()
	switch errCode {
	case "session_superseded", "session_expired", "waiting_room_required", "waiting_room_queued", "freebuff_update_required":
		return body, true
	default:
		return body, false
	}
}

func freebuffUsageDetailOrEstimate(baseModel string, requestPayload, responsePayload []byte) usage.Detail {
	detail := helps.ParseOpenAIUsage(responsePayload)
	if freebuffHasUsage(detail) {
		return detail
	}
	return freebuffEstimateUsage(baseModel, requestPayload, responsePayload)
}

func freebuffHasUsage(detail usage.Detail) bool {
	return detail.InputTokens > 0 ||
		detail.OutputTokens > 0 ||
		detail.ReasoningTokens > 0 ||
		detail.CachedTokens > 0 ||
		detail.TotalTokens > 0
}

func freebuffEstimateUsage(baseModel string, requestPayload, responsePayload []byte) usage.Detail {
	inputTokens := freebuffEstimateInputTokens(baseModel, requestPayload)
	segments := make([]string, 0, 8)
	root := gjson.ParseBytes(responsePayload)
	choices := root.Get("choices")
	if choices.Exists() && choices.IsArray() {
		choices.ForEach(func(_, choice gjson.Result) bool {
			message := choice.Get("message")
			if content := message.Get("content"); content.Exists() && content.Type == gjson.String {
				if text := strings.TrimSpace(content.String()); text != "" {
					segments = append(segments, text)
				}
			}
			toolCalls := message.Get("tool_calls")
			if toolCalls.Exists() && toolCalls.IsArray() {
				toolCalls.ForEach(func(_, tc gjson.Result) bool {
					if name := strings.TrimSpace(tc.Get("function.name").String()); name != "" {
						segments = append(segments, name)
					}
					if args := strings.TrimSpace(tc.Get("function.arguments").String()); args != "" {
						segments = append(segments, args)
					}
					return true
				})
			}
			return true
		})
	}
	return freebuffEstimateUsageFromSegments(baseModel, inputTokens, segments)
}

func freebuffEstimateInputTokens(baseModel string, requestPayload []byte) int64 {
	enc, err := helps.TokenizerForModel(baseModel)
	if err != nil {
		return 0
	}
	count, err := helps.CountOpenAIChatTokens(enc, requestPayload)
	if err != nil {
		return 0
	}
	return count
}

func freebuffEstimateUsageFromSegments(baseModel string, inputTokens int64, segments []string) usage.Detail {
	outputTokens := freebuffEstimateOutputTokens(baseModel, segments)
	return usage.Detail{
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		TotalTokens:  inputTokens + outputTokens,
	}
}

func freebuffEstimateOutputTokens(baseModel string, segments []string) int64 {
	if len(segments) == 0 {
		return 0
	}
	enc, err := helps.TokenizerForModel(baseModel)
	if err != nil {
		return 0
	}
	return freebuffCountSegments(enc, segments)
}

func freebuffCountSegments(enc tokenizer.Codec, segments []string) int64 {
	if enc == nil || len(segments) == 0 {
		return 0
	}
	joined := strings.TrimSpace(strings.Join(segments, "\n"))
	if joined == "" {
		return 0
	}
	count, err := enc.Count(joined)
	if err != nil {
		return 0
	}
	return int64(count)
}

func freebuffCollectOutputSegments(line []byte, segments *[]string) {
	payload := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
	if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) || !gjson.ValidBytes(payload) {
		return
	}
	root := gjson.ParseBytes(payload)
	if delta := root.Get("choices.0.delta.content"); delta.Exists() {
		if text := strings.TrimSpace(delta.String()); text != "" {
			*segments = append(*segments, text)
		}
	}
	if tcArray := root.Get("choices.0.delta.tool_calls"); tcArray.Exists() && tcArray.IsArray() {
		tcArray.ForEach(func(_, tc gjson.Result) bool {
			if name := strings.TrimSpace(tc.Get("function.name").String()); name != "" {
				*segments = append(*segments, name)
			}
			if args := strings.TrimSpace(tc.Get("function.arguments").String()); args != "" {
				*segments = append(*segments, args)
			}
			return true
		})
	}
}

// applyFreebuffHeaders sets the required HTTP headers for Freebuff requests.
func applyFreebuffHeaders(req *http.Request, authToken string, stream bool) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+authToken)
	req.Header.Set("User-Agent", freebuffUserAgent)
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	} else {
		req.Header.Set("Accept", "application/json")
	}
}

// freebuffToolCallDelta holds accumulated state for a single tool call being streamed.
type freebuffToolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`
	} `json:"function,omitempty"`
}

// assembleFreebuffSSEResponse reassembles an SSE stream into a single OpenAI chat completion response.
// It preserves tool_calls deltas and extracts real usage if present in the stream.
func assembleFreebuffSSEResponse(sseData []byte, model string) []byte {
	var contentParts []string
	var finishReason string
	toolCalls := make(map[int]*freebuffToolCallDelta) // index -> accumulated tool call
	var streamUsage map[string]any

	lines := bytes.Split(sseData, []byte("\n"))
	for _, line := range lines {
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		jsonStr := bytes.TrimSpace(line[6:])
		if bytes.Equal(jsonStr, []byte("[DONE]")) {
			continue
		}
		parsed := gjson.ParseBytes(jsonStr)

		// Accumulate content deltas.
		if delta := parsed.Get("choices.0.delta.content"); delta.Exists() && delta.String() != "" {
			contentParts = append(contentParts, delta.String())
		}

		// Accumulate tool_calls deltas.
		tcArray := parsed.Get("choices.0.delta.tool_calls")
		if tcArray.Exists() && tcArray.IsArray() {
			for _, tc := range tcArray.Array() {
				idx := int(tc.Get("index").Int())
				existing, ok := toolCalls[idx]
				if !ok {
					existing = &freebuffToolCallDelta{Index: idx}
					toolCalls[idx] = existing
				}
				if id := tc.Get("id").String(); id != "" {
					existing.ID = id
				}
				if t := tc.Get("type").String(); t != "" {
					existing.Type = t
				}
				if name := tc.Get("function.name").String(); name != "" {
					existing.Function.Name = name
				}
				existing.Function.Arguments += tc.Get("function.arguments").String()
			}
		}

		// Extract finish_reason.
		if fr := parsed.Get("choices.0.finish_reason"); fr.Exists() && fr.String() != "" {
			finishReason = fr.String()
		}

		// Extract usage from the stream if the upstream provides it.
		if u := parsed.Get("usage"); u.Exists() && u.IsObject() {
			var uMap map[string]any
			if errU := json.Unmarshal([]byte(u.Raw), &uMap); errU == nil {
				streamUsage = uMap
			}
		}
	}

	if finishReason == "" {
		finishReason = "stop"
	}

	message := map[string]any{
		"role":    "assistant",
		"content": strings.Join(contentParts, ""),
	}

	// Build sorted tool_calls array if any were accumulated.
	if len(toolCalls) > 0 {
		sortedCalls := make([]any, 0, len(toolCalls))
		for i := 0; i < len(toolCalls); i++ {
			if tc, ok := toolCalls[i]; ok {
				sortedCalls = append(sortedCalls, tc)
			}
		}
		// Append any remaining non-contiguous indices.
		if len(sortedCalls) < len(toolCalls) {
			for _, tc := range toolCalls {
				found := false
				for _, s := range sortedCalls {
					if s.(*freebuffToolCallDelta).Index == tc.Index {
						found = true
						break
					}
				}
				if !found {
					sortedCalls = append(sortedCalls, tc)
				}
			}
		}
		message["tool_calls"] = sortedCalls
		// When tool calls are present, content is often empty/null.
		if message["content"] == "" {
			message["content"] = nil
		}
	}

	result := map[string]any{
		"id":      fmt.Sprintf("freebuff-%d", time.Now().UnixMilli()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]any{
			{
				"index":         0,
				"message":       message,
				"finish_reason": finishReason,
			},
		},
	}
	// Only include usage if the stream actually provided it.
	if streamUsage != nil {
		result["usage"] = streamUsage
	}
	data, _ := json.Marshal(result)
	return data
}

// freebuffRandomAlphanumeric generates a random alphanumeric string of the given length.
func freebuffRandomAlphanumeric(n int) string {
	const chars = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, n)
	for i := range b {
		b[i] = chars[rand.Intn(len(chars))]
	}
	return string(b)
}
