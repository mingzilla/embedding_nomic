using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

/// <summary>
/// Simple timer utility for tracking operation durations and identifying bottlenecks.
/// Usage:
///   using (var timer = new SimpleTimer("Processing batch"))
///   {
///       // do operation 1
///       timer.Track("sql_read");
///       // do operation 2
///       timer.Track("processing");
///       // do operation 3
///       timer.Track("embedding");
///       // do operation 4
///       timer.Track("db_write_bulk");
///   } // Automatically logs timing summary on dispose
/// </summary>
public class SimpleTimer : IDisposable
{
    private readonly string _message;
    private readonly Stopwatch _stopwatch;
    private readonly List<(string name, long durationMs)> _tracks;
    private long _lastTrackMs;

    public SimpleTimer(string message = null)
    {
        _message = message;
        _stopwatch = new Stopwatch();
        _tracks = new List<(string name, long durationMs)>();
        _lastTrackMs = 0;
    }

    /// <summary>
    /// Gets the total elapsed time in milliseconds
    /// </summary>
    public long ElapsedMs => _stopwatch.ElapsedMilliseconds;

    /// <summary>
    /// Starts the timer. Call this to begin tracking.
    /// </summary>
    public SimpleTimer Start()
    {
        if (_message != null)
        {
            Console.WriteLine(_message);
        }
        _stopwatch.Start();
        _lastTrackMs = 0;
        return this;
    }

    /// <summary>
    /// Records a timing checkpoint for the given operation name
    /// </summary>
    public void Track(string name)
    {
        long now = _stopwatch.ElapsedMilliseconds;
        long duration = now - _lastTrackMs;
        _tracks.Add((name, duration));
        _lastTrackMs = now;
    }

    /// <summary>
    /// Gets timing summary as a list of tuples (step name, duration in ms, total duration in ms)
    /// </summary>
    public List<(string step, long durationMs)> GetTimingSummary()
    {
        var result = new List<(string step, long durationMs)>(_tracks);
        result.Add(("total", ElapsedMs));
        return result;
    }

    /// <summary>
    /// Constructs a single-line timer message
    /// Example: "Timer: [step1: 123ms], [step2: 456ms], TOTAL: 579ms"
    /// </summary>
    private string ConstructOneRowMessage()
    {
        var totalMs = ElapsedMs;
        var timingParts = _tracks.Select(t => $"[{t.name}: {t.durationMs}ms]");
        var totalStr = $"TOTAL: {totalMs}ms";

        if (!_tracks.Any())
        {
            return $"Timer: {totalStr}";
        }
        else
        {
            return "Timer: " + string.Join(", ", timingParts) + $", {totalStr}";
        }
    }

    /// <summary>
    /// Constructs a multi-line tabular timer message with percentages and bottleneck identification
    /// Example:
    ///   04.28% - [sql_read: 276ms]
    ///   00.01% - [processing: 1ms]
    ///   93.75% - [embedding: 6036ms]    - (BOTTLENECK)
    ///   01.95% - [db_write_bulk: 126ms]
    ///   TOTAL: 6438ms
    /// </summary>
    private string ConstructTabularMessage()
    {
        var totalMs = ElapsedMs;

        // Find bottleneck (step with max duration)
        string bottleneckName = null;
        if (_tracks.Any())
        {
            bottleneckName = _tracks.OrderByDescending(t => t.durationMs).First().name;
        }

        // Calculate max width for step labels
        int maxLabelWidth = 0;
        if (_tracks.Any())
        {
            maxLabelWidth = _tracks.Max(t => $"[{t.name}: {t.durationMs}ms]".Length);
        }
        maxLabelWidth = Math.Max(maxLabelWidth, $"TOTAL: {totalMs}ms".Length);

        var lines = new List<string> { "" }; // Start with empty line for break

        // Add each step with percentage
        foreach (var (name, durationMs) in _tracks)
        {
            double percentage = totalMs > 0 ? (durationMs / (double)totalMs * 100) : 0;
            string label = $"[{name}: {durationMs}ms]";
            bool isBottleneck = (name == bottleneckName);

            // Format: "  05.30% - [step_name: 123ms]" with optional bottleneck marker
            string line = $"  {percentage:00.00}% - {label.PadRight(maxLabelWidth)}";
            if (isBottleneck)
            {
                line += " - (BOTTLENECK)";
            }
            lines.Add(line);
        }

        // Add total with time conversion
        string totalLabel = $"TOTAL: {totalMs}ms";
        double minutes = totalMs / 60000.0;
        string timeSuffix = minutes >= 1 ? $" ({minutes:F1} minutes)" : "";
        lines.Add($"  {totalLabel}{timeSuffix}");

        return string.Join(Environment.NewLine, lines);
    }

    /// <summary>
    /// Stops the timer and logs the timing summary.
    /// Uses one-row format for < 3 items, tabular format otherwise.
    /// </summary>
    public void Dispose()
    {
        _stopwatch.Stop();

        // Use one-row format for < 3 items, tabular format otherwise
        string logMessage = _tracks.Count < 3
            ? ConstructOneRowMessage()
            : ConstructTabularMessage();

        Console.WriteLine(logMessage);
    }
}
