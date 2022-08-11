use aleph_alpha_client::{Client, Error, Prompt, Sampling, TaskCompletion};
use wiremock::{
    matchers::{body_json_string, header, method, path},
    Mock, MockServer, ResponseTemplate,
};

#[tokio::test]
async fn completion_with_luminous_base() {
    // Given

    // Start a background HTTP server on a random local part
    let mock_server = MockServer::start().await;

    let answer = r#"{"model_version":"2021-12","completions":[{"completion":"\n","finish_reason":"maximum_tokens"}]}"#;
    let body = r#"{
        "model": "luminous-base",
        "prompt": [{"type": "text", "data": "Hello,"}],
        "maximum_tokens": 1
    }"#;

    Mock::given(method("POST"))
        .and(path("/complete"))
        .and(header("Authorization", "Bearer dummy-token"))
        .and(header("Content-Type", "application/json"))
        .and(body_json_string(body))
        .respond_with(ResponseTemplate::new(200).set_body_string(answer))
        // Mounting the mock on the mock server - it's now effective!
        .mount(&mock_server)
        .await;

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_text("Hello,"),
        maximum_tokens: 1,
        sampling: Sampling::MOST_LIKELY,
    };

    let model = "luminous-base";

    let client = Client::with_base_url(mock_server.uri(), "dummy-token").unwrap();
    let response = client.complete(model, &task).await.unwrap();
    let actual = response.completion;

    // Then
    assert_eq!("\n", actual)
}

/// If we open too many requests at once, we may trigger rate limmiting. We want this scenario to be
/// easily detectible by the user, so he/she/it can start sending requests slower.
#[tokio::test]
async fn detect_rate_limmiting() {
    // Given

    // Start a background HTTP server on a random local part
    let mock_server = MockServer::start().await;

    let answer = r#"Too many requests"#;
    let body = r#"{
        "model": "luminous-base",
        "prompt": [{"type": "text", "data": "Hello,"}],
        "maximum_tokens": 1
    }"#;

    Mock::given(method("POST"))
        .and(path("/complete"))
        .and(header("Authorization", "Bearer dummy-token"))
        .and(header("Content-Type", "application/json"))
        .and(body_json_string(body))
        .respond_with(ResponseTemplate::new(429).set_body_string(answer))
        // Mounting the mock on the mock server - it's now effective!
        .mount(&mock_server)
        .await;

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_text("Hello,"),
        maximum_tokens: 1,
        sampling: Sampling::MOST_LIKELY,
    };

    let model = "luminous-base";

    let client = Client::with_base_url(mock_server.uri(), "dummy-token").unwrap();
    let error = client.complete(model, &task).await.unwrap_err();

    assert!(matches!(error, Error::TooManyRequests));
}

/// Even if we do not open too many requests at once ourselfes, the API may just be busy. We also
/// want this scenario to be easily detectable by users.
#[tokio::test]
async fn detect_queue_full() {
    // Given

    // Start a background HTTP server on a random local part
    let mock_server = MockServer::start().await;

    let answer = r#"{
        "error":"Sorry we had to reject your request because we could not guarantee to finish it in
            a reasonable timeframe. This specific model is very busy at this moment. Try again later
            or use another model.",
        "code":"QUEUE_FULL"
    }"#;
    let body = r#"{
        "model": "luminous-base",
        "prompt": [{"type": "text", "data": "Hello,"}],
        "maximum_tokens": 1
    }"#;

    Mock::given(method("POST"))
        .and(path("/complete"))
        .and(header("Authorization", "Bearer dummy-token"))
        .and(header("Content-Type", "application/json"))
        .and(body_json_string(body))
        .respond_with(ResponseTemplate::new(503).set_body_string(answer))
        // Mounting the mock on the mock server - it's now effective!
        .mount(&mock_server)
        .await;

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_text("Hello,"),
        maximum_tokens: 1,
        sampling: Sampling::MOST_LIKELY,
    };

    let model = "luminous-base";

    let client = Client::with_base_url(mock_server.uri(), "dummy-token").unwrap();
    let error = client.complete(model, &task).await.unwrap_err();

    assert!(matches!(error, Error::Busy));
}
