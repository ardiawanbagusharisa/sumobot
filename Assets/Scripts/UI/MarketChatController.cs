using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using SumoMultiplayer;
using SumoServices;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Functional local-first chat for the Market placeholder. Messages are stored
/// in a shared JSON-lines file, so two builds on the same machine can chat while
/// testing with different UGS/local profiles.
/// </summary>
public sealed class MarketChatController : MonoBehaviour
{
    private TMP_InputField inputField;
    private Button sendButton;
    private TMP_Text messageLog;
    private float nextPollAt;
    private long lastFileLength = -1;
    private DateTime lastWriteTimeUtc;
    private bool initialized;

    public void Initialize()
    {
        if (initialized)
            return;

        initialized = true;
        inputField = GetComponentsInChildren<TMP_InputField>(true).FirstOrDefault();
        sendButton = GetComponentsInChildren<Button>(true)
            .FirstOrDefault(button => button.name == "ButtonSend");

        BuildMessageLog();
        if (sendButton != null)
            sendButton.onClick.AddListener(SendCurrentMessage);
        if (inputField != null)
        {
            inputField.characterLimit = 240;
            inputField.onSubmit.AddListener(_ => SendCurrentMessage());
        }

        RefreshNow();
    }

    private void OnEnable()
    {
        if (!initialized)
            Initialize();
        RefreshNow();
    }

    private void Update()
    {
        if (Time.unscaledTime < nextPollAt)
            return;

        nextPollAt = Time.unscaledTime + 0.75f;
        FileInfo file = LocalMarketChatStore.GetFileInfo();
        long length = file.Exists ? file.Length : 0;
        DateTime writeTime = file.Exists ? file.LastWriteTimeUtc : DateTime.MinValue;
        if (length != lastFileLength || writeTime != lastWriteTimeUtc)
            RefreshNow();
    }

    public void SetDraft(string text)
    {
        Initialize();
        if (inputField == null)
            return;

        inputField.text = text ?? string.Empty;
        inputField.ActivateInputField();
        inputField.MoveTextEnd(false);
    }

    public void RefreshNow()
    {
        Initialize();
        IReadOnlyList<MarketChatMessage> messages = LocalMarketChatStore.LoadRecent(14);
        if (messageLog != null)
        {
            messageLog.SetText(messages.Count == 0
                ? "No messages yet. Ask about an item or say hello."
                : string.Join("\n", messages.Select(FormatMessage)));
        }

        FileInfo file = LocalMarketChatStore.GetFileInfo();
        file.Refresh();
        lastFileLength = file.Exists ? file.Length : 0;
        lastWriteTimeUtc = file.Exists ? file.LastWriteTimeUtc : DateTime.MinValue;
    }

    private void SendCurrentMessage()
    {
        string text = inputField?.text?.Trim();
        if (string.IsNullOrWhiteSpace(text))
            return;

        PlayerAccount account = GameServices.Auth?.Current;
        string senderId = !string.IsNullOrWhiteSpace(account?.PlayerId)
            ? account.PlayerId
            : OnlineBattleSession.ProfileName;
        string senderName = !string.IsNullOrWhiteSpace(account?.DisplayName)
            ? account.DisplayName
            : OnlineBattleSession.ProfileName;

        if (!LocalMarketChatStore.TryAppend(new MarketChatMessage
            {
                Id = Guid.NewGuid().ToString("N"),
                SenderId = senderId,
                SenderName = senderName,
                Text = text.Replace("\r", " ").Replace("\n", " "),
                TimestampUtc = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture)
            }, out string error))
        {
            Logger.Warning($"[MarketChat] Could not send message: {error}");
            return;
        }

        inputField.text = string.Empty;
        inputField.ActivateInputField();
        RefreshNow();
    }

    private string FormatMessage(MarketChatMessage message)
    {
        string timestamp = DateTime.TryParse(
            message.TimestampUtc,
            CultureInfo.InvariantCulture,
            DateTimeStyles.RoundtripKind,
            out DateTime parsed)
                ? parsed.ToLocalTime().ToString("HH:mm")
                : "--:--";
        string name = string.IsNullOrWhiteSpace(message.SenderName) ? "Player" : message.SenderName;
        return $"{timestamp}  {name}: {message.Text}";
    }

    private void BuildMessageLog()
    {
        TMP_FontAsset font = GetComponentsInChildren<TMP_Text>(true)
            .Select(text => text.font)
            .FirstOrDefault(candidate => candidate != null);

        var background = new GameObject(
            "RuntimeChatLog",
            typeof(RectTransform),
            typeof(CanvasRenderer),
            typeof(Image));
        background.transform.SetParent(transform, false);
        RectTransform backgroundRect = background.GetComponent<RectTransform>();
        backgroundRect.anchorMin = new Vector2(0.06f, 0.23f);
        backgroundRect.anchorMax = new Vector2(0.94f, 0.84f);
        backgroundRect.offsetMin = Vector2.zero;
        backgroundRect.offsetMax = Vector2.zero;
        background.GetComponent<Image>().color = new Color(0.96f, 0.97f, 1f, 0.98f);

        var textObject = new GameObject(
            "Messages",
            typeof(RectTransform),
            typeof(CanvasRenderer),
            typeof(TextMeshProUGUI));
        textObject.transform.SetParent(background.transform, false);
        RectTransform textRect = textObject.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = new Vector2(16f, 12f);
        textRect.offsetMax = new Vector2(-16f, -12f);

        TextMeshProUGUI text = textObject.GetComponent<TextMeshProUGUI>();
        text.font = font;
        text.fontSize = 16f;
        text.color = new Color(0.12f, 0.14f, 0.2f, 1f);
        text.alignment = TextAlignmentOptions.BottomLeft;
        text.textWrappingMode = TextWrappingModes.Normal;
        text.overflowMode = TextOverflowModes.Truncate;
        text.richText = false;
        messageLog = text;
    }
}

[Serializable]
public sealed class MarketChatMessage
{
    public string Id;
    public string SenderId;
    public string SenderName;
    public string Text;
    public string TimestampUtc;
}

internal static class LocalMarketChatStore
{
    private const string FileName = "market-chat.jsonl";

    private static string FilePath => Path.Combine(
        Application.persistentDataPath,
        "Chat",
        FileName);

    public static FileInfo GetFileInfo() => new(FilePath);

    public static IReadOnlyList<MarketChatMessage> LoadRecent(int count)
    {
        if (!File.Exists(FilePath))
            return Array.Empty<MarketChatMessage>();

        try
        {
            using var stream = new FileStream(FilePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            using var reader = new StreamReader(stream);
            var messages = new List<MarketChatMessage>();
            while (reader.ReadLine() is { } line)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                try
                {
                    MarketChatMessage message = JsonConvert.DeserializeObject<MarketChatMessage>(line);
                    if (message != null && !string.IsNullOrWhiteSpace(message.Text))
                        messages.Add(message);
                }
                catch (JsonException)
                {
                    // Ignore an incomplete/corrupt line instead of breaking chat history.
                }
            }

            return messages.Skip(Mathf.Max(0, messages.Count - Mathf.Max(1, count))).ToList();
        }
        catch (IOException ex)
        {
            Logger.Warning($"[MarketChat] Could not read chat history: {ex.Message}");
            return Array.Empty<MarketChatMessage>();
        }
    }

    public static bool TryAppend(MarketChatMessage message, out string error)
    {
        error = string.Empty;
        try
        {
            string directory = Path.GetDirectoryName(FilePath);
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);

            string line = JsonConvert.SerializeObject(message, Formatting.None);
            using var stream = new FileStream(FilePath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite);
            using var writer = new StreamWriter(stream);
            writer.WriteLine(line);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            error = ex.Message;
            return false;
        }
    }
}
