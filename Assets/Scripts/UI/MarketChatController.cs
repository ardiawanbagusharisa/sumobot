using System;
using System.Collections;
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
    private ScrollRect messageScroll;
    private ScrollRect peopleScroll;
    private TMP_InputField peopleSearch;
    private readonly List<GameObject> peopleRows = new();
    private TMP_Text onlineHeader;
    private Coroutine scrollToLatest;
    private float nextPollAt;
    private long lastFileLength = -1;
    private DateTime lastWriteTimeUtc;
    private bool initialized;

    public void Initialize()
    {
        if (initialized)
            return;

        initialized = true;
        // The placeholder has two TMP inputs: a search field at the top and the
        // actual message composer along the bottom. Bind by layout so renamed
        // scene objects still work.
        inputField = GetComponentsInChildren<TMP_InputField>(true)
            .OrderBy(field => field.GetComponent<RectTransform>().anchorMax.y)
            .ThenByDescending(field => field.GetComponent<RectTransform>().rect.width)
            .FirstOrDefault();
        sendButton = GetComponentsInChildren<Button>(true)
            .FirstOrDefault(button => button.name == "ButtonSend");

        BindMessageLog();
        BindPeopleList();
        if (sendButton != null)
            sendButton.onClick.AddListener(SendCurrentMessage);
        if (inputField != null)
        {
            inputField.characterLimit = 240;
            inputField.onSubmit.AddListener(_ => SendCurrentMessage());
        }
        if (peopleSearch != null)
            peopleSearch.onValueChanged.AddListener(_ => RefreshPeopleList());

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
        else
            RefreshPeopleList();
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
            ResizeMessageContent();
        }
        RefreshPeopleList();

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

    private void BindMessageLog()
    {
        messageScroll = GetComponentsInChildren<ScrollRect>(true)
            .OrderByDescending(scroll => scroll.GetComponent<RectTransform>().rect.width)
            .FirstOrDefault();
        if (messageScroll == null || messageScroll.content == null)
        {
            Logger.Warning("[MarketChat] The existing message Scroll View was not found.");
            return;
        }

        messageLog = messageScroll.content.GetComponentsInChildren<TMP_Text>(true)
            .OrderByDescending(text => text.rectTransform.rect.width * text.rectTransform.rect.height)
            .FirstOrDefault();
        if (messageLog == null)
        {
            var textObject = new GameObject(
                "Messages",
                typeof(RectTransform),
                typeof(CanvasRenderer),
                typeof(TextMeshProUGUI));
            textObject.transform.SetParent(messageScroll.content, false);
            messageLog = textObject.GetComponent<TextMeshProUGUI>();
            messageLog.font = GetComponentsInChildren<TMP_Text>(true)
                .Select(text => text.font)
                .FirstOrDefault(candidate => candidate != null);
        }

        RectTransform textRect = messageLog.rectTransform;
        textRect.anchorMin = new Vector2(0f, 1f);
        textRect.anchorMax = new Vector2(1f, 1f);
        textRect.pivot = new Vector2(0.5f, 1f);
        textRect.anchoredPosition = new Vector2(0f, -8f);
        textRect.sizeDelta = new Vector2(-24f, Mathf.Max(100f, textRect.sizeDelta.y));
        messageLog.fontSize = Mathf.Max(15f, messageLog.fontSize);
        messageLog.color = Color.white;
        messageLog.alignment = TextAlignmentOptions.TopLeft;
        messageLog.textWrappingMode = TextWrappingModes.Normal;
        messageLog.overflowMode = TextOverflowModes.Overflow;
        messageLog.richText = false;
    }

    private void BindPeopleList()
    {
        peopleScroll = GetComponentsInChildren<ScrollRect>(true)
            .FirstOrDefault(scroll => scroll.name == "Scroll View (2)");
        if (peopleScroll?.content == null)
        {
            Logger.Warning("[MarketChat] The existing player list was not found.");
            return;
        }

        foreach (Transform child in peopleScroll.content)
        {
            if (child.name.StartsWith("ChatUser Group", StringComparison.Ordinal))
            {
                if (onlineHeader == null)
                    onlineHeader = child.GetComponentInChildren<TMP_Text>(true);
                else
                    child.gameObject.SetActive(false);
            }
            else if (child.name.StartsWith("ChatUser", StringComparison.Ordinal))
            {
                peopleRows.Add(child.gameObject);
            }
        }

        peopleSearch = GetComponentsInChildren<TMP_InputField>(true)
            .FirstOrDefault(field => field.name == "InputField (TMP) Search");
    }

    private void RefreshPeopleList()
    {
        if (peopleRows.Count == 0)
            return;

        IReadOnlyList<MarketOnlinePlayer> online = LocalMarketPresence.GetOnlinePlayers();
        if (onlineHeader != null)
            onlineHeader.SetText($"Online here ({online.Count})");

        // The first existing row is the public channel; the remaining rows are
        // the scene's player placeholders. They are not fake users or DMs.
        TMP_Text publicLabel = peopleRows[0].GetComponentInChildren<TMP_Text>(true);
        if (publicLabel != null)
            publicLabel.SetText("Community chat");
        peopleRows[0].SetActive(true);

        string filter = peopleSearch?.text?.Trim();
        List<MarketOnlinePlayer> visible = online
            .Where(player => string.IsNullOrEmpty(filter) ||
                player.DisplayName.IndexOf(filter, StringComparison.OrdinalIgnoreCase) >= 0)
            .ToList();

        while (peopleRows.Count - 1 < visible.Count)
        {
            GameObject row = Instantiate(peopleRows[peopleRows.Count - 1], peopleScroll.content);
            row.name = $"ChatUser Online {peopleRows.Count}";
            peopleRows.Add(row);
        }

        for (int index = 1; index < peopleRows.Count; index++)
        {
            GameObject row = peopleRows[index];
            int playerIndex = index - 1;
            bool show = playerIndex < visible.Count;
            row.SetActive(show);
            if (!show)
                continue;

            MarketOnlinePlayer player = visible[playerIndex];
            TMP_Text label = row.GetComponentInChildren<TMP_Text>(true);
            if (label != null)
                label.SetText(player.IsSelf ? $"{player.DisplayName} (you)" : player.DisplayName);

            Transform indicator = row.transform.Find("Image (1)");
            if (indicator != null && indicator.TryGetComponent(out Image status))
                status.color = new Color(0f, 0.85f, 0.38f, 1f);

            Button button = row.GetComponentInChildren<Button>(true);
            if (button != null)
            {
                button.onClick.RemoveAllListeners();
                string mention = player.DisplayName;
                button.onClick.AddListener(() => SetDraft($"@{mention} "));
            }
        }

        LayoutRebuilder.ForceRebuildLayoutImmediate(peopleScroll.content);
    }

    private void ResizeMessageContent()
    {
        if (messageLog == null || messageScroll == null || messageScroll.content == null)
            return;

        Canvas.ForceUpdateCanvases();
        float viewportHeight = messageScroll.viewport != null
            ? messageScroll.viewport.rect.height
            : messageScroll.GetComponent<RectTransform>().rect.height;
        float height = Mathf.Max(viewportHeight, messageLog.preferredHeight + 24f);
        // The authored scroll content uses a VerticalLayoutGroup and
        // ContentSizeFitter. Grow its existing ChatPanel so the fitter, scrollbar
        // and text agree on the actual message height.
        RectTransform chatPanel = messageLog.rectTransform.parent as RectTransform;
        if (chatPanel != null)
            chatPanel.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, height);
        messageLog.rectTransform.SetSizeWithCurrentAnchors(
            RectTransform.Axis.Vertical,
            height - 16f);
        LayoutRebuilder.ForceRebuildLayoutImmediate(messageScroll.content);
        if (isActiveAndEnabled)
        {
            if (scrollToLatest != null)
                StopCoroutine(scrollToLatest);
            scrollToLatest = StartCoroutine(ScrollToLatestAfterLayout());
        }
    }

    private IEnumerator ScrollToLatestAfterLayout()
    {
        yield return null;
        Canvas.ForceUpdateCanvases();
        LayoutRebuilder.ForceRebuildLayoutImmediate(messageScroll.content);
        messageScroll.StopMovement();
        messageScroll.verticalNormalizedPosition = 0f;
        if (messageScroll.verticalScrollbar != null)
            messageScroll.verticalScrollbar.value = 0f;
        scrollToLatest = null;
    }
}

[Serializable]
internal sealed class MarketOnlinePlayer
{
    public string SessionId;
    public string PlayerId;
    public string DisplayName;
    [JsonIgnore] public bool IsSelf;
}

/// <summary>
/// A short-lived presence lease for the same-machine chat transport. Each game
/// process writes its own file, including while another scene is open, and an
/// unexpected process exit disappears from the list after a few seconds.
/// </summary>
internal sealed class LocalMarketPresence : MonoBehaviour
{
    private const float HeartbeatSeconds = 2f;
    private const int LeaseSeconds = 8;
    private static LocalMarketPresence instance;
    private string sessionId;
    private string ownFile;
    private float nextHeartbeat;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void Bootstrap()
    {
        if (instance != null)
            return;

        var root = new GameObject("LocalMarketPresence");
        DontDestroyOnLoad(root);
        root.AddComponent<LocalMarketPresence>();
    }

    private void Awake()
    {
        if (instance != null && instance != this)
        {
            Destroy(gameObject);
            return;
        }

        instance = this;
        sessionId = Guid.NewGuid().ToString("N");
        ownFile = Path.Combine(DirectoryPath, $"presence-{sessionId}.json");
    }

    private void Update()
    {
        if (Time.unscaledTime < nextHeartbeat)
            return;

        nextHeartbeat = Time.unscaledTime + HeartbeatSeconds;
        try
        {
            Directory.CreateDirectory(DirectoryPath);
            PlayerAccount account = GameServices.Auth?.Current;
            string profile = OnlineBattleSession.ProfileName;
            var player = new MarketOnlinePlayer
            {
                SessionId = sessionId,
                PlayerId = string.IsNullOrWhiteSpace(account?.PlayerId) ? profile : account.PlayerId,
                DisplayName = string.IsNullOrWhiteSpace(account?.DisplayName) ||
                    string.Equals(account.DisplayName, "Player", StringComparison.OrdinalIgnoreCase)
                        ? profile
                        : account.DisplayName
            };
            using var stream = new FileStream(ownFile, FileMode.Create, FileAccess.Write, FileShare.ReadWrite);
            using var writer = new StreamWriter(stream);
            writer.Write(JsonConvert.SerializeObject(player));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Logger.Warning($"[MarketChat] Could not update local presence: {ex.Message}");
        }
    }

    private void OnApplicationQuit() => RemoveOwnLease();

    private void OnDestroy()
    {
        if (instance != this)
            return;

        RemoveOwnLease();
        instance = null;
    }

    private void RemoveOwnLease()
    {
        try
        {
            if (!string.IsNullOrEmpty(ownFile) && File.Exists(ownFile))
                File.Delete(ownFile);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // The lease expires even if shutdown races another client's read.
        }
    }

    private static string DirectoryPath => Path.Combine(Application.persistentDataPath, "Chat", "Presence");

    public static IReadOnlyList<MarketOnlinePlayer> GetOnlinePlayers()
    {
        if (!Directory.Exists(DirectoryPath))
            return Array.Empty<MarketOnlinePlayer>();

        var players = new List<MarketOnlinePlayer>();
        try
        {
            foreach (string file in Directory.EnumerateFiles(DirectoryPath, "presence-*.json"))
            {
                if (DateTime.UtcNow - File.GetLastWriteTimeUtc(file) > TimeSpan.FromSeconds(LeaseSeconds))
                    continue;

                try
                {
                    using var stream = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
                    using var reader = new StreamReader(stream);
                    MarketOnlinePlayer player = JsonConvert.DeserializeObject<MarketOnlinePlayer>(reader.ReadToEnd());
                    if (player == null || string.IsNullOrWhiteSpace(player.DisplayName))
                        continue;

                    player.IsSelf = instance != null && player.SessionId == instance.sessionId;
                    players.Add(player);
                }
                catch (Exception ex) when (ex is IOException or JsonException)
                {
                    // A peer may be halfway through a heartbeat. Try next poll.
                }
            }
        }
        catch (IOException)
        {
            return players;
        }

        return players.OrderByDescending(player => player.IsSelf)
            .ThenBy(player => player.DisplayName, StringComparer.OrdinalIgnoreCase)
            .ToList();
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
