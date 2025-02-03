//! Usage sample
//!
//! ```no_run
//! use aleph_alpha_client::{Client, TaskCompletion, How};
//!
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() {
//!     // Authenticate against API. Fetches token.
//!     let client = Client::from_env().unwrap();
//!
//!     // Name of the model we we want to use. Large models give usually better answer, but are also
//!     // more costly.
//!     let model = "luminous-base";
//!
//!     // The task we want to perform. Here we want to continue the sentence: "An apple a day ..."
//!     let task = TaskCompletion::from_text("An apple a day");
//!     
//!     // Retrieve the answer from the API
//!     let response = client.completion(&task, model, &How::default()).await.unwrap();
//!
//!     // Print entire sentence with completion
//!     println!("An apple a day{}", response.completion);
//! }
//! ```

mod chat;
mod completion;
mod detokenization;
mod explanation;
mod http;
mod image_preprocessing;
mod prompt;
mod semantic_embedding;
mod stream;
mod tokenization;
use dotenvy::dotenv;
use futures_util::Stream;
use http::HttpClient;
use semantic_embedding::{BatchSemanticEmbeddingOutput, SemanticEmbeddingOutput};
use std::env;
use std::{pin::Pin, time::Duration};
use tokenizers::Tokenizer;

pub use self::{
    chat::{ChatEvent, ChatStreamChunk},
    chat::{ChatOutput, Message, TaskChat},
    completion::{CompletionEvent, CompletionSummary, StreamChunk, StreamSummary},
    completion::{CompletionOutput, Sampling, Stopping, TaskCompletion},
    detokenization::{DetokenizationOutput, TaskDetokenization},
    explanation::{
        Explanation, ExplanationOutput, Granularity, ImageScore, ItemExplanation,
        PromptGranularity, TaskExplanation, TextScore,
    },
    http::{Error, Job, Task},
    prompt::{Modality, Prompt},
    semantic_embedding::{
        SemanticRepresentation, TaskBatchSemanticEmbedding, TaskSemanticEmbedding,
    },
    stream::{StreamJob, StreamTask},
    tokenization::{TaskTokenization, TokenizationOutput},
};

/// Execute Jobs against the Aleph Alpha API
pub struct Client {
    /// This client does all the work of sending the requests and talking to the AA API. The only
    /// additional knowledge added by this layer is that it knows about the individual jobs which
    /// can be executed, which allows for an alternative non generic interface which might produce
    /// easier to read code for the end user in many use cases.
    http_client: HttpClient,
}

impl Client {
    /// A new instance of an Aleph Alpha client helping you interact with the Aleph Alpha API.
    ///
    /// Setting the token to None allows specifying it on a per request basis.
    /// You may want to only use request based authentication and skip default authentication. This
    /// is useful if writing an application which invokes the client on behalf of many different
    /// users. Having neither request, nor default authentication is considered a bug and will cause
    /// a panic.
    pub fn new(host: impl Into<String>, api_token: Option<String>) -> Result<Self, Error> {
        let http_client = HttpClient::new(host.into(), api_token)?;
        Ok(Self { http_client })
    }

    /// A client instance that always uses the same token for all requests.
    pub fn with_auth(host: impl Into<String>, api_token: impl Into<String>) -> Result<Self, Error> {
        Self::new(host, Some(api_token.into()))
    }

    pub fn from_env() -> Result<Self, Error> {
        let _ = dotenv();
        let api_token = env::var("PHARIA_AI_TOKEN").unwrap();
        let inference_url = env::var("INFERENCE_URL").unwrap();
        Self::with_auth(inference_url, api_token)
    }

    /// Execute a task with the aleph alpha API and fetch its result.
    ///
    /// ```no_run
    /// use aleph_alpha_client::{Client, How, TaskCompletion, Error};
    ///
    /// async fn print_completion() -> Result<(), Error> {
    ///     // Authenticate against API. Fetches token.
    ///     let client = Client::from_env()?;
    ///
    ///     // Name of the model we we want to use. Large models give usually better answer, but are
    ///     // also slower and more costly.
    ///     let model = "luminous-base";
    ///
    ///     // The task we want to perform. Here we want to continue the sentence: "An apple a day
    ///     // ..."
    ///     let task = TaskCompletion::from_text("An apple a day");
    ///
    ///     // Retrieve answer from API
    ///     let response = client.execute(model, &task, &How::default()).await?;
    ///
    ///     // Print entire sentence with completion
    ///     println!("An apple a day{}", response.completion);
    ///     Ok(())
    /// }
    /// ```
    #[deprecated = "Please use output_of instead."]
    pub async fn execute<T: Task>(
        &self,
        model: &str,
        task: &T,
        how: &How,
    ) -> Result<T::Output, Error> {
        self.output_of(&task.with_model(model), how).await
    }

    /// Execute any task with the aleph alpha API and fetch its result. This is most useful in
    /// generic code then you want to execute arbitrary task types. Otherwise prefer methods taking
    /// concrete tasks like [`Self::completion`] for improved readability.
    pub async fn output_of<T: Job>(&self, task: &T, how: &How) -> Result<T::Output, Error> {
        self.http_client.output_of(task, how).await
    }

    /// An embedding trying to capture the semantic meaning of a text. Cosine similarity can be used
    /// find out how well two texts (or multimodal prompts) match. Useful for search usecases.
    ///
    /// See the example for [`cosine_similarity`].
    pub async fn semantic_embedding(
        &self,
        task: &TaskSemanticEmbedding<'_>,
        how: &How,
    ) -> Result<SemanticEmbeddingOutput, Error> {
        self.http_client.output_of(task, how).await
    }

    /// An batch of embeddings trying to capture the semantic meaning of a text.
    pub async fn batch_semantic_embedding(
        &self,
        task: &TaskBatchSemanticEmbedding<'_>,
        how: &How,
    ) -> Result<BatchSemanticEmbeddingOutput, Error> {
        self.http_client.output_of(task, how).await
    }

    /// Instruct a model served by the aleph alpha API to continue writing a piece of text (or
    /// multimodal document).
    ///
    /// ```no_run
    /// use aleph_alpha_client::{Client, How, TaskCompletion, Task, Error};
    ///
    /// async fn print_completion() -> Result<(), Error> {
    ///     // Authenticate against API. Fetches token.
    ///     let client = Client::from_env()?;
    ///
    ///     // Name of the model we we want to use. Large models give usually better answer, but are
    ///     // also slower and more costly.
    ///     let model = "luminous-base";
    ///
    ///     // The task we want to perform. Here we want to continue the sentence: "An apple a day
    ///     // ..."
    ///     let task = TaskCompletion::from_text("An apple a day");
    ///
    ///     // Retrieve answer from API
    ///     let response = client.completion(&task, model, &How::default()).await?;
    ///
    ///     // Print entire sentence with completion
    ///     println!("An apple a day{}", response.completion);
    ///     Ok(())
    /// }
    /// ```
    pub async fn completion(
        &self,
        task: &TaskCompletion<'_>,
        model: &str,
        how: &How,
    ) -> Result<CompletionOutput, Error> {
        self.http_client
            .output_of(&Task::with_model(task, model), how)
            .await
    }

    /// Instruct a model served by the aleph alpha API to continue writing a piece of text.
    /// Stream the response as a series of events.
    ///
    /// ```no_run
    /// use aleph_alpha_client::{Client, How, TaskCompletion, Error, CompletionEvent};
    /// use futures_util::StreamExt;
    ///
    /// async fn print_stream_completion() -> Result<(), Error> {
    ///     // Authenticate against API. Fetches token.
    ///     let client = Client::from_env()?;
    ///
    ///     // Name of the model we we want to use. Large models give usually better answer, but are
    ///     // also slower and more costly.
    ///     let model = "luminous-base";
    ///
    ///     // The task we want to perform. Here we want to continue the sentence: "An apple a day
    ///     // ..."
    ///     let task = TaskCompletion::from_text("An apple a day");
    ///
    ///     // Retrieve stream from API
    ///     let mut stream = client.stream_completion(&task, model, &How::default()).await?;
    ///     while let Some(Ok(event)) = stream.next().await {
    ///         if let CompletionEvent::StreamChunk(chunk) = event {
    ///             println!("{}", chunk.completion);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn stream_completion(
        &self,
        task: &TaskCompletion<'_>,
        model: &str,
        how: &How,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionEvent, Error>> + Send>>, Error> {
        self.http_client
            .stream_output_of(&Task::with_model(task, model), how)
            .await
    }

    /// Send a chat message to a model.
    /// ```no_run
    /// use aleph_alpha_client::{Client, How, TaskChat, Error, Message};
    ///
    /// async fn print_chat() -> Result<(), Error> {
    ///     // Authenticate against API. Fetches token.
    ///     let client = Client::from_env()?;
    ///
    ///     // Name of a model that supports chat.
    ///     let model = "pharia-1-llm-7b-control";
    ///
    ///     // Create a chat task with a user message.
    ///     let message = Message::user("Hello, how are you?");
    ///     let task = TaskChat::with_message(message);
    ///
    ///     // Send the message to the model.
    ///     let response = client.chat(&task, model, &How::default()).await?;
    ///
    ///     // Print the model response
    ///     println!("{}", response.message.content);
    ///     Ok(())
    /// }
    /// ```
    pub async fn chat(
        &self,
        task: &TaskChat<'_>,
        model: &str,
        how: &How,
    ) -> Result<ChatOutput, Error> {
        self.http_client
            .output_of(&Task::with_model(task, model), how)
            .await
    }

    /// Send a chat message to a model. Stream the response as a series of events.
    /// ```no_run
    /// use aleph_alpha_client::{Client, How, TaskChat, Error, Message};
    /// use futures_util::StreamExt;
    ///
    /// async fn print_stream_chat() -> Result<(), Error> {
    ///     // Authenticate against API. Fetches token.
    ///     let client = Client::from_env()?;
    ///
    ///     // Name of a model that supports chat.
    ///     let model = "pharia-1-llm-7b-control";
    ///
    ///     // Create a chat task with a user message.
    ///     let message = Message::user("Hello, how are you?");
    ///     let task = TaskChat::with_message(message);
    ///
    ///     // Send the message to the model.
    ///     let mut stream = client.stream_chat(&task, model, &How::default()).await?;
    ///     while let Some(Ok(event)) = stream.next().await {
    ///          println!("{}", event.delta.content);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn stream_chat(
        &self,
        task: &TaskChat<'_>,
        model: &str,
        how: &How,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamChunk, Error>> + Send>>, Error> {
        self.http_client
            .stream_output_of(&StreamTask::with_model(task, model), how)
            .await
    }

    /// Returns an explanation given a prompt and a target (typically generated
    /// by a previous completion request). The explanation describes how individual parts
    /// of the prompt influenced the target.
    ///
    /// ```no_run
    /// use aleph_alpha_client::{Client, How, TaskCompletion, Task, Error, Granularity, TaskExplanation, Stopping, Prompt, Sampling};
    ///
    /// async fn print_explanation() -> Result<(), Error> {
    ///     let client = Client::from_env()?;
    ///
    ///     // Name of the model we we want to use. Large models give usually better answer, but are
    ///     // also slower and more costly.
    ///     let model = "luminous-base";
    ///
    ///     // input for the completion
    ///     let prompt = Prompt::from_text("An apple a day");
    ///
    ///     let task = TaskCompletion {
    ///         prompt: prompt.clone(),
    ///         stopping: Stopping::from_maximum_tokens(10),
    ///         sampling: Sampling::MOST_LIKELY,
    ///         special_tokens: false,
    ///     };
    ///     let response = client.completion(&task, model, &How::default()).await?;
    ///
    ///     let task = TaskExplanation {
    ///         prompt: prompt,               // same input as for completion
    ///         target: &response.completion,  // output of completion
    ///         granularity: Granularity::default(),
    ///     };
    ///     let response = client.explanation(&task, model, &How::default()).await?;
    ///
    ///     dbg!(&response);
    ///     Ok(())
    /// }
    /// ```
    pub async fn explanation(
        &self,
        task: &TaskExplanation<'_>,
        model: &str,
        how: &How,
    ) -> Result<ExplanationOutput, Error> {
        self.http_client
            .output_of(&task.with_model(model), how)
            .await
    }

    /// Tokenize a prompt for a specific model.
    ///
    /// ```no_run
    /// use aleph_alpha_client::{Client, Error, How, TaskTokenization};
    ///
    /// async fn tokenize() -> Result<(), Error> {
    ///     let client = Client::from_env()?;
    ///
    ///     // Name of the model for which we want to tokenize text.
    ///     let model = "luminous-base";
    ///
    ///     // Text prompt to be tokenized.
    ///     let prompt = "An apple a day";
    ///
    ///     let task = TaskTokenization {
    ///         prompt,
    ///         tokens: true,       // return text-tokens
    ///         token_ids: true,    // return numeric token-ids
    ///     };
    ///     let responses = client.tokenize(&task, model, &How::default()).await?;
    ///
    ///     dbg!(&responses);
    ///     Ok(())
    /// }
    /// ```
    pub async fn tokenize(
        &self,
        task: &TaskTokenization<'_>,
        model: &str,
        how: &How,
    ) -> Result<TokenizationOutput, Error> {
        self.http_client
            .output_of(&task.with_model(model), how)
            .await
    }

    /// Detokenize a list of token ids into a string.
    ///
    /// ```no_run
    /// use aleph_alpha_client::{Client, Error, How, TaskDetokenization};
    ///
    /// async fn detokenize() -> Result<(), Error> {
    ///     let client = Client::from_env()?;
    ///
    ///     // Specify the name of the model whose tokenizer was used to generate the input token ids.
    ///     let model = "luminous-base";
    ///
    ///     // Token ids to convert into text.
    ///     let token_ids: Vec<u32> = vec![556, 48741, 247, 2983];
    ///
    ///     let task = TaskDetokenization {
    ///         token_ids: &token_ids,
    ///     };
    ///     let responses = client.detokenize(&task, model, &How::default()).await?;
    ///
    ///     dbg!(&responses);
    ///     Ok(())
    /// }
    /// ```
    pub async fn detokenize(
        &self,
        task: &TaskDetokenization<'_>,
        model: &str,
        how: &How,
    ) -> Result<DetokenizationOutput, Error> {
        self.http_client
            .output_of(&task.with_model(model), how)
            .await
    }

    pub async fn tokenizer_by_model(
        &self,
        model: &str,
        api_token: Option<String>,
    ) -> Result<Tokenizer, Error> {
        self.http_client.tokenizer_by_model(model, api_token).await
    }
}

/// Controls of how to execute a task
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct How {
    /// The be-nice flag is used to reduce load for the models you intend to use.
    /// This is commonly used if you are conducting experiments
    /// or trying things out that create a large load on the aleph-alpha-api
    /// and you do not want to increase queue time for other users too much.
    ///
    /// (!) This increases how often you get a `Busy` response.
    pub be_nice: bool,

    /// The maximum duration of a request before the client cancels the request. This is not passed on
    /// to the server but only handled by the client locally, i.e. the client will not wait longer than
    /// this duration for a response.
    pub client_timeout: Duration,

    /// API token used to authenticate the request, overwrites the default token provided on setup
    /// Default token may not provide the tracking or permission that is wanted for the request
    pub api_token: Option<String>,

    /// Optionally pass a trace context to propagate tracing information through distributed systems.
    pub trace_context: Option<TraceContext>,
}

/// Trace context that is propagated through HTTP headers to enable distributed tracing.
///
/// Compliant with https://www.w3.org/TR/trace-context-2/#design-overview, which standardizes how
/// context information is sent and modified between services.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TraceContext {
    /// traceparent describes the position of the incoming request in its trace graph in a portable,
    /// fixed-length format. Its design focuses on fast parsing. Every tracing tool MUST properly set
    /// traceparent even when it only relies on vendor-specific information in tracestate
    pub traceparent: String,

    /// tracestate extends traceparent with vendor-specific data represented by a set of name/value
    /// pairs. Storing information in tracestate is optional.
    pub tracestate: String,
}

impl TraceContext {
    pub fn new(traceparent: impl Into<String>, tracestate: impl Into<String>) -> Self {
        Self {
            traceparent: traceparent.into(),
            tracestate: tracestate.into(),
        }
    }
}

impl Default for How {
    fn default() -> Self {
        // the aleph-alpha-api cancels request after 5 minute
        let api_timeout = Duration::from_secs(300);
        Self {
            be_nice: Default::default(),
            // on the client side a request can take longer in case of network errors
            // therefore by default we wait slightly longer
            client_timeout: api_timeout + Duration::from_secs(5),
            api_token: None,
            trace_context: None,
        }
    }
}

/// Intended to compare embeddings.
///
/// ```no_run
/// use aleph_alpha_client::{
///     Client, Prompt, TaskSemanticEmbedding, cosine_similarity, SemanticRepresentation, How
/// };
///
/// async fn semantic_search_with_luminous_base(client: &Client) {
///     // Given
///     let robot_fact = Prompt::from_text(
///         "A robot is a machine—especially one programmable by a computer—capable of carrying out a \
///         complex series of actions automatically.",
///     );
///     let pizza_fact = Prompt::from_text(
///         "Pizza (Italian: [ˈpittsa], Neapolitan: [ˈpittsə]) is a dish of Italian origin consisting \
///         of a usually round, flat base of leavened wheat-based dough topped with tomatoes, cheese, \
///         and often various other ingredients (such as various types of sausage, anchovies, \
///         mushrooms, onions, olives, vegetables, meat, ham, etc.), which is then baked at a high \
///         temperature, traditionally in a wood-fired oven.",
///     );
///     let query = Prompt::from_text("What is Pizza?");
///     let how = How::default();
///     
///     // When
///     let robot_embedding_task = TaskSemanticEmbedding {
///         prompt: robot_fact,
///         representation: SemanticRepresentation::Document,
///         compress_to_size: Some(128),
///     };
///     let robot_embedding = client.semantic_embedding(
///         &robot_embedding_task,
///         &how,
///     ).await.unwrap().embedding;
///     
///     let pizza_embedding_task = TaskSemanticEmbedding {
///         prompt: pizza_fact,
///         representation: SemanticRepresentation::Document,
///         compress_to_size: Some(128),
///     };
///     let pizza_embedding = client.semantic_embedding(
///         &pizza_embedding_task,
///         &how,
///     ).await.unwrap().embedding;
///     
///     let query_embedding_task = TaskSemanticEmbedding {
///         prompt: query,
///         representation: SemanticRepresentation::Query,
///         compress_to_size: Some(128),
///     };
///     let query_embedding = client.semantic_embedding(
///         &query_embedding_task,
///         &how,
///     ).await.unwrap().embedding;
///     let similarity_pizza = cosine_similarity(&query_embedding, &pizza_embedding);
///     println!("similarity pizza: {similarity_pizza}");
///     let similarity_robot = cosine_similarity(&query_embedding, &robot_embedding);
///     println!("similarity robot: {similarity_robot}");
///     
///     // Then
///     
///     // The fact about pizza should be more relevant to the "What is Pizza?" question than a fact
///     // about robots.
///     assert!(similarity_pizza > similarity_robot);
/// }
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let ab: f32 = a.iter().zip(b).map(|(a, b)| a * b).sum();
    let aa: f32 = a.iter().map(|a| a * a).sum();
    let bb: f32 = b.iter().map(|b| b * b).sum();
    let prod_len = (aa * bb).sqrt();
    ab / prod_len
}

#[cfg(test)]
mod tests {
    use crate::Prompt;

    #[test]
    fn ability_to_generate_prompt_in_local_function() {
        fn local_function() -> Prompt<'static> {
            Prompt::from_text(String::from("My test prompt"))
        }

        assert_eq!(Prompt::from_text("My test prompt"), local_function())
    }
}
