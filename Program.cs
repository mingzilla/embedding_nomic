using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

public class Program
{
    // Configuration
    const string text = "MARINE AND GENERAL MUTUAL LIFE ASSURANCE SOCIETY";
    const string HuggingFaceRepo = "nomic-ai/nomic-embed-text-v1.5";
    const string ModelFileName = "nomic-embed-text-v1.5.onnx";
    const string VocabFileName = "vocab.txt";
    const string ModelPath = "./model/onnx";
    const int EmbeddingDimension = 128;
    const int MaxSequenceLength = 8192;

    public static async Task Main(string[] args)
    {
        Console.WriteLine("Starting ONNX embedding test...\n");

        // Ensure model directory exists
        Directory.CreateDirectory(ModelPath);
        string modelFilePath = Path.Combine(ModelPath, ModelFileName);
        string vocabFilePath = Path.Combine(ModelPath, VocabFileName);

        // Download files if needed
        await DownloadModelIfNeeded(HuggingFaceRepo, "onnx/" + ModelFileName, modelFilePath);
        await DownloadModelIfNeeded(HuggingFaceRepo, VocabFileName, vocabFilePath);

        // Generate embedding
        Console.WriteLine($"\nGenerating embedding for text: \"{text}\"");
        float[] embedding = await GenerateEmbedding(modelFilePath, vocabFilePath, text, EmbeddingDimension);

        Console.WriteLine($"\nEmbedding generated successfully!");
        Console.WriteLine($"Dimension: {embedding.Length}");
        Console.WriteLine($"First 10 values: [{string.Join(", ", embedding.Take(10).Select(v => v.ToString("F6")))}]");
        Console.WriteLine($"Last 10 values: [{string.Join(", ", embedding.Skip(embedding.Length - 10).Select(v => v.ToString("F6")))}]");
    }

    public static async Task<float[]> GenerateEmbedding(string modelPath, string vocabPath, string text, int embeddingDimension)
    {
        Console.WriteLine("Loading tokenizer...");

        // Create a BertTokenizer from the vocabulary file.
        using var vocabStream = File.OpenRead(vocabPath);
        var tokenizer = BertTokenizer.Create(vocabStream);

        // Add task prefix for nomic models
        string prefixedText = "search_document: " + text;

        // Tokenize
        Console.WriteLine("Tokenizing text...");
        var tokenIds = tokenizer.EncodeToIds(prefixedText);
        var inputIds = tokenIds.Select(id => (long)id).ToArray();
        var attentionMask = Enumerable.Repeat(1L, inputIds.Length).ToArray();
        var tokenTypeIds = Enumerable.Repeat(0L, inputIds.Length).ToArray();

        Console.WriteLine($"Tokenized to {inputIds.Length} tokens");

        // The old code had an unnecessary `await Task.CompletedTask;`, which is removed.

        // Load ONNX model and attempt to use CUDA provider
        Console.WriteLine("Loading ONNX model...");
        var sessionOptions = new SessionOptions();
        try
        {
            Console.WriteLine("Attempting to use CUDA execution provider...");
            sessionOptions.AppendExecutionProvider_CUDA(0);
            Console.WriteLine("CUDA execution provider successfully loaded.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Failed to load CUDA execution provider. ONNX Runtime will fall back to CPU. Error: {ex.Message}");
        }

        using var session = new InferenceSession(modelPath, sessionOptions);

        Console.WriteLine("Running inference...");

        // Prepare input tensors
        var inputIdsTensor = new DenseTensor<long>(inputIds, new[] { 1, inputIds.Length });
        var attentionMaskTensor = new DenseTensor<long>(attentionMask, new[] { 1, attentionMask.Length });
        var tokenTypeIdsTensor = new DenseTensor<long>(tokenTypeIds, new[] { 1, tokenTypeIds.Length });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
            NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIdsTensor)
        };

        // Run inference
        using var results = session.Run(inputs);

        // Extract the last_hidden_state tensor, which contains the embeddings for each token.
        var lastHiddenStateTensor = results.First(r => r.Name == "last_hidden_state").AsTensor<float>();

        // --- Perform Mean Pooling ---
        // Get the dimensions of the tensor and the data as a span.
        var dimensions = lastHiddenStateTensor.Dimensions;
        var sequenceLength = (int)dimensions[1];
        var hiddenSize = (int)dimensions[2];
        var lastHiddenStateData = lastHiddenStateTensor.ToDenseTensor().Buffer.Span;

        // Create an array to hold the pooled embedding.
        var pooledEmbedding = new float[hiddenSize];
        
        // Count the number of active tokens (where attention_mask is 1).
        var activeTokenCount = attentionMask.Count(m => m == 1);

        if (activeTokenCount > 0)
        {
            // Iterate through each token's embedding in the sequence.
            for (int i = 0; i < sequenceLength; i++)
            {
                // Only consider tokens that are not padding.
                if (attentionMask[i] == 1)
                {
                    // Get a slice of the data representing the current token's embedding.
                    var tokenEmbedding = lastHiddenStateData.Slice(i * hiddenSize, hiddenSize);
                    
                    // Add the current token's embedding to the pooled embedding.
                    for (int j = 0; j < hiddenSize; j++)
                    {
                        pooledEmbedding[j] += tokenEmbedding[j];
                    }
                }
            }

            // Divide the summed embeddings by the number of active tokens to get the mean.
            for (int j = 0; j < hiddenSize; j++)
            {
                pooledEmbedding[j] /= activeTokenCount;
            }
        }

        float[] embedding = pooledEmbedding;

        Console.WriteLine($"Original embedding dimension: {embedding.Length}");

        // Truncate to desired dimension if needed
        if (embedding.Length > embeddingDimension)
        {
            Console.WriteLine($"Truncating to dimension: {embeddingDimension}");
            float[] truncated = new float[embeddingDimension];
            Array.Copy(embedding, truncated, embeddingDimension);
            return truncated;
        }

        return embedding;
    }

    public static async Task DownloadModelIfNeeded(string repo, string fileName, string destPath)
    {
        if (File.Exists(destPath))
        {
            Console.WriteLine($"File already exists at: {destPath}");
            return;
        }

        Console.WriteLine($"Downloading '{fileName}' from '{repo}'...");
        string url = $"https://huggingface.co/{repo}/resolve/main/{fileName}";

        using var client = new HttpClient();
        client.Timeout = TimeSpan.FromMinutes(30);

        using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        long? totalBytes = response.Content.Headers.ContentLength;

        using var contentStream = await response.Content.ReadAsStreamAsync();
        using var fileStream = new FileStream(destPath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true);

        var buffer = new byte[8192];
        long downloadedBytes = 0;
        int bytesRead;

        while ((bytesRead = await contentStream.ReadAsync(buffer, 0, buffer.Length)) > 0)
        {
            await fileStream.WriteAsync(buffer, 0, bytesRead);
            downloadedBytes += bytesRead;

            if (totalBytes.HasValue)
            {
                double percentage = (double)downloadedBytes / totalBytes.Value * 100;
                Console.Write($"\rDownloading: {downloadedBytes / 1024.0 / 1024.0:F2} MB / {totalBytes.Value / 1024.0 / 1024.0:F2} MB ({percentage:F2}%)");
            }
            else
            {
                Console.Write($"\rDownloading: {downloadedBytes / 1024.0 / 1024.0:F2} MB");
            }
        }

        Console.WriteLine("\nDownload complete!");
    }
}
