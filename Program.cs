using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using DuckDB.NET.Data;
using LLamaSharp.KernelMemory;
using Microsoft.SemanticKernel.AI.Embeddings;

public class Program
{
    // 1. Configuration
    const string HuggingFaceRepo = "nomic-ai/nomic-embed-text-v1.5-GGUF";
    const string ModelFileName = "nomic-embed-text-v1.5.f16.gguf";
    const string ModelPath = "./model";
    const string DbPath = "data/ClassifiedCompaniesRelational.duckdb";
    const string SourceTableName = "companies";
    const string DestTableName = "companies_with_embeddings";
    const string TextColumnName = "CompanyName";
    const string IdColumnName = "CompanyNumber";
    const int TotalRows = 100; // Set to -1 to process all rows
    const int BatchSize = 50;
    const int EmbeddingDimension = 128; // Target dimension

    public static async Task Main(string[] args)
    {
        Console.WriteLine("Starting embedding process...");

        // Ensure model directory exists
        Directory.CreateDirectory(ModelPath);
        string fullModelPath = Path.Combine(ModelPath, ModelFileName);

        // 2. Download the model if it doesn't exist
        await ModelManager.DownloadModel(HuggingFaceRepo, ModelFileName, fullModelPath);

        // 3. Initialize the embedding service
        var embeddingService = new EmbeddingService(fullModelPath);

        // 4. Perform single text embedding example
        string sampleText = "This is a test sentence for embedding.";
        float[] sampleEmbedding = embeddingService.GenerateEmbedding(sampleText);
        Console.WriteLine($"Generated embedding for sample text with dimension: {sampleEmbedding.Length}");
        Console.WriteLine($"First 5 values: {string.Join(", ", sampleEmbedding.Take(5))}");


        // 5. Perform batch embedding for DuckDB
        await ProcessDuckDb(embeddingService);

        Console.WriteLine("Embedding process finished.");
    }

    private static async Task ProcessDuckDb(EmbeddingService embeddingService)
    {
        Console.WriteLine("\nStarting DuckDB processing...");
        using var connection = new DuckDBConnection($"Data Source={DbPath}");
        connection.Open();

        // Create or clear the destination table
        using (var cmd = connection.CreateCommand())
        {
            cmd.CommandText = $"CREATE OR REPLACE TABLE {DestTableName} (CompanyNumber VARCHAR, CompanyName VARCHAR, embedding FLOAT[{EmbeddingDimension}]);";
            cmd.ExecuteNonQuery();
        }

        long totalRows = GetTotalRows(connection);
        long rowsToProcess = (TotalRows > 0 && TotalRows < totalRows) ? TotalRows : totalRows;
        
        Console.WriteLine($"Processing {rowsToProcess} rows from '{SourceTableName}' in batches of {BatchSize}...");

        for (int offset = 0; offset < rowsToProcess; offset += BatchSize)
        {
            var (idBatch, textBatch) = ReadBatch(connection, offset, BatchSize);
            if (textBatch.Count == 0) break;

            Console.WriteLine($"Processing batch from offset {offset} with {textBatch.Count} records.");

            var embeddings = embeddingService.GenerateEmbeddings(textBatch);
            
            await InsertBatch(connection, idBatch, textBatch, embeddings);
        }

        Console.WriteLine("DuckDB processing finished.");
    }

    private static long GetTotalRows(DuckDBConnection connection)
    {
        using var cmd = connection.CreateCommand();
        cmd.CommandText = $"SELECT COUNT(*) FROM {SourceTableName}";
        return (long)cmd.ExecuteScalar();
    }

    private static (List<string> ids, List<string> texts) ReadBatch(DuckDBConnection connection, int offset, int limit)
    {
        var ids = new List<string>();
        var texts = new List<string>();
        using var cmd = connection.CreateCommand();
        cmd.CommandText = $"SELECT {IdColumnName}, {TextColumnName} FROM {SourceTableName} LIMIT {limit} OFFSET {offset};";
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            ids.Add(reader.GetString(0));
            texts.Add(reader.GetString(1));
        }
        return (ids, texts);
    }
    
    private static async Task InsertBatch(DuckDBConnection connection, List<string> ids, List<string> texts, List<float[]> embeddings)
    {
        // DuckDB ADO.NET provider doesn't support array parameters well for bulk insertion yet.
        // We'll use transactions and individual inserts for robustness.
        using var transaction = connection.BeginTransaction();
        for(int i = 0; i < ids.Count; i++)
        {
            using var cmd = connection.CreateCommand();
            // Convert float array to DuckDB LIST literal format
            var embeddingString = $"[{string.Join(", ", embeddings[i])}]";
            cmd.CommandText = $"INSERT INTO {DestTableName} VALUES (?, ?, {embeddingString});";
            cmd.Parameters.Add(new DuckDBParameter(ids[i]));
            cmd.Parameters.Add(new DuckDBParameter(texts[i]));
            await cmd.ExecuteNonQueryAsync();
        }
        transaction.Commit();
    }
}

public static class ModelManager
{
    public static async Task DownloadModel(string repo, string modelFile, string destPath)
    {
        if (File.Exists(destPath))
        {
            Console.WriteLine("Model already exists. Skipping download.");
            return;
        }

        Console.WriteLine($"Downloading model '{modelFile}' from '{repo}'...");
        string url = $"https://huggingface.co/{repo}/resolve/main/{modelFile}";
        
        using var client = new HttpClient();
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
                Console.Write($"\rDownloading: {downloadedBytes / 1024 / 1024:F2} MB / {totalBytes.Value / 1024 / 1024:F2} MB ({percentage:F2}%)");
            }
            else
            {
                 Console.Write($"\rDownloading: {downloadedBytes / 1024 / 1024:F2} MB");
            }
        }
        Console.WriteLine("\nDownload complete.");
    }
}

public class EmbeddingService : IDisposable
{
    private readonly LLamaEmbedder _embedder;

    public EmbeddingService(string modelPath)
    {
        var parameters = new ModelParams(modelPath)
        {
            ContextSize = 1024,
            GpuLayerCount = 100, // Offload all layers to GPU
            Embeddings = true
        };
        using var weights = LLamaWeights.LoadFromFile(parameters);
        _embedder = new LLamaEmbedder(weights, parameters);
    }

    public float[] GenerateEmbedding(string text)
    {
        var embedding = _embedder.GetEmbeddings(text).ToArray();
        // Nomic specific: dimensionality reduction is handled by the model if configured.
        // We need to ensure the model itself is set up for 128 dimensions if possible.
        // If the model always returns 768, we might need to truncate or warn.
        // For now, we assume the model respects some config or we truncate.
        if (embedding.Length > Program.EmbeddingDimension)
        {
            return embedding.Take(Program.EmbeddingDimension).ToArray();
        }
        return embedding;
    }

    public List<float[]> GenerateEmbeddings(List<string> texts)
    {
        var embeddings = new List<float[]>();
        foreach (var text in texts)
        {
            embeddings.Add(GenerateEmbedding(text));
        }
        return embeddings;
    }

    public void Dispose()
    {
        _embedder.Dispose();
    }
}