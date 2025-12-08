using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using LLama;
using LLama.Common;

public class Program
{
    // Configuration
    const string text = "MARINE AND GENERAL MUTUAL LIFE ASSURANCE SOCIETY";
    const string HuggingFaceRepo = "nomic-ai/nomic-embed-text-v1.5-GGUF";
    const string ModelFileName = "nomic-embed-text-v1.5.f16.gguf";
    const string ModelPath = "./model";
    const int EmbeddingDimension = 128;

    public static async Task Main(string[] args)
    {
        Console.WriteLine("Starting simple embedding test...\n");

        // Ensure model directory exists
        Directory.CreateDirectory(ModelPath);
        string fullModelPath = Path.Combine(ModelPath, ModelFileName);

        // Download the model if it doesn't exist
        await DownloadModelIfNeeded(HuggingFaceRepo, ModelFileName, fullModelPath);

        // Generate embedding for "hello"
        Console.WriteLine($"\nGenerating embedding for text: \"{text}\"");
        float[] embedding = await GenerateEmbedding(fullModelPath, text, EmbeddingDimension);

        Console.WriteLine($"\nEmbedding generated successfully!");
        Console.WriteLine($"Dimension: {embedding.Length}");
        Console.WriteLine($"First 10 values: [{string.Join(", ", embedding.Take(10).Select(v => v.ToString("F6")))}]");
        Console.WriteLine($"Last 10 values: [{string.Join(", ", embedding.Skip(embedding.Length - 10).Select(v => v.ToString("F6")))}]");
    }

    public static async Task<float[]> GenerateEmbedding(string modelPath, string text, int embeddingDimension)
    {
        Console.WriteLine("Loading model...");

        var parameters = new ModelParams(modelPath)
        {
            ContextSize = 1024,
            GpuLayerCount = 100, // Use GPU
            Embeddings = true
        };

        using var weights = LLamaWeights.LoadFromFile(parameters);
        using var embedder = new LLamaEmbedder(weights, parameters);

        Console.WriteLine("Model loaded. Generating embedding...");

        // Nomic models require a prefix for optimal performance
        string prefixedText = "search_document: " + text;

        // GetEmbeddings returns Task<IReadOnlyList<float[]>>, await and get first result
        var embeddingsList = await embedder.GetEmbeddings(prefixedText);
        float[] embedding = embeddingsList[0];

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

    public static async Task DownloadModelIfNeeded(string repo, string modelFile, string destPath)
    {
        if (File.Exists(destPath))
        {
            Console.WriteLine($"Model already exists at: {destPath}");
            return;
        }

        Console.WriteLine($"Downloading model '{modelFile}' from '{repo}'...");
        string url = $"https://huggingface.co/{repo}/resolve/main/{modelFile}";

        using var client = new HttpClient();
        client.Timeout = TimeSpan.FromMinutes(30); // Long timeout for large model files

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
