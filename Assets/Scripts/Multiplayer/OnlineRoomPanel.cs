using System;
using System.Collections.Generic;
using SumoInput;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace SumoMultiplayer
{
    /// <summary>
    /// Runtime room browser used by the existing MainMenu multiplayer screen.
    /// Building it at runtime keeps the networking package out of scene YAML and
    /// lets both landscape/portrait variants use the same implementation.
    /// </summary>
    public sealed class OnlineRoomPanel : MonoBehaviour
    {
        public readonly struct RoomEntry
        {
            public readonly string Id;
            public readonly string Name;
            public readonly int PlayerCount;
            public readonly int MaxPlayers;

            public RoomEntry(string id, string name, int playerCount, int maxPlayers)
            {
                Id = id;
                Name = name;
                PlayerCount = playerCount;
                MaxPlayers = maxPlayers;
            }
        }

        private const int MaxVisibleRooms = 4;

        private TMP_FontAsset font;
        private TMP_Text statusText;
        private TMP_Text inputModeText;
        private TMP_Text closeButtonText;
        private TMP_Text lobbyStatusText;
        private TMP_Text readyButtonText;
        private TMP_Text roomsTitle;
        private Button inputModeButton;
        private Button createButton;
        private Button refreshButton;
        private Button closeButton;
        private Button readyButton;
        private readonly List<Button> roomButtons = new();
        private readonly List<TMP_Text> roomLabels = new();
        private readonly List<RoomEntry> displayedRooms = new();
        private Action createRequested;
        private Action refreshRequested;
        private Action closeRequested;
        private Action readyRequested;
        private Action<string> joinRequested;
        private InputType selectedInputType = InputType.UI;
        private bool waiting;

        public bool IsVisible => gameObject.activeSelf;
        public InputType SelectedInputType => selectedInputType;

        public static OnlineRoomPanel Create(
            Canvas canvas,
            TMP_FontAsset font,
            Action createRequested,
            Action refreshRequested,
            Action closeRequested,
            Action readyRequested,
            Action<string> joinRequested)
        {
            var overlay = new GameObject(
                "OnlineRoomBrowser",
                typeof(RectTransform),
                typeof(CanvasRenderer),
                typeof(Image));
            overlay.transform.SetParent(canvas.transform, false);

            RectTransform rect = overlay.GetComponent<RectTransform>();
            rect.anchorMin = Vector2.zero;
            rect.anchorMax = Vector2.one;
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;
            overlay.GetComponent<Image>().color = new Color(0.05f, 0.07f, 0.16f, 0.78f);

            OnlineRoomPanel panel = overlay.AddComponent<OnlineRoomPanel>();
            panel.font = font;
            panel.createRequested = createRequested;
            panel.refreshRequested = refreshRequested;
            panel.closeRequested = closeRequested;
            panel.readyRequested = readyRequested;
            panel.joinRequested = joinRequested;
            panel.Build();
            overlay.SetActive(false);
            return panel;
        }

        public void Show(InputType initialInputType)
        {
            selectedInputType = initialInputType == InputType.LiveCommand
                ? InputType.LiveCommand
                : InputType.UI;
            UpdateInputModeLabel();
            transform.SetAsLastSibling();
            gameObject.SetActive(true);
        }

        public void Hide() => gameObject.SetActive(false);

        public void SetStatus(string status)
        {
            if (statusText != null)
                statusText.SetText(status ?? string.Empty);
        }

        public void SetBusy(bool busy)
        {
            if (createButton != null) createButton.interactable = !busy;
            if (refreshButton != null) refreshButton.interactable = !busy;
            if (inputModeButton != null) inputModeButton.interactable = !busy;
            foreach (Button button in roomButtons)
                button.interactable = !busy && button.gameObject.activeSelf;
        }

        public void SetWaiting(bool waiting)
        {
            this.waiting = waiting;
            if (readyButton != null) readyButton.gameObject.SetActive(waiting);
            if (lobbyStatusText != null) lobbyStatusText.gameObject.SetActive(waiting);
            if (roomsTitle != null) roomsTitle.gameObject.SetActive(!waiting);
            if (createButton != null) createButton.gameObject.SetActive(!waiting);
            if (refreshButton != null) refreshButton.gameObject.SetActive(!waiting);
            for (int index = 0; index < roomButtons.Count; index++)
                roomButtons[index].gameObject.SetActive(!waiting && index < displayedRooms.Count);
            SetBusy(waiting);
            if (closeButtonText != null)
                closeButtonText.SetText(waiting ? "Leave room" : "Close");
            if (waiting)
                SetLobbyState(false, false, 0, false);
        }

        public void SetLobbyState(bool localReady, bool opponentReady, int seconds, bool opponentConnected)
        {
            if (!waiting)
                return;

            if (readyButton != null)
                readyButton.interactable = opponentConnected && !localReady;
            if (readyButtonText != null)
                readyButtonText.SetText(localReady ? "Ready ✓" : "Ready now");
            if (lobbyStatusText != null)
                lobbyStatusText.SetText(opponentConnected
                    ? $"You: {(localReady ? "ready" : "not ready")}     Opponent: {(opponentReady ? "ready" : "not ready")}\nBattle loads in {seconds}s, or sooner when both are ready."
                    : "Waiting for an opponent to join...");
        }

        public void SetRooms(IReadOnlyList<RoomEntry> rooms)
        {
            displayedRooms.Clear();
            if (rooms != null)
                displayedRooms.AddRange(rooms);
            int count = Mathf.Min(displayedRooms.Count, MaxVisibleRooms);
            for (int index = 0; index < roomButtons.Count; index++)
            {
                bool visible = !waiting && index < count;
                Button button = roomButtons[index];
                button.gameObject.SetActive(visible);
                button.onClick.RemoveAllListeners();
                if (index >= count)
                    continue;

                RoomEntry entry = displayedRooms[index];
                roomLabels[index].SetText(
                    $"{entry.Name}    {entry.PlayerCount}/{entry.MaxPlayers}    JOIN");
                string roomId = entry.Id;
                button.onClick.AddListener(() => joinRequested?.Invoke(roomId));
            }

            if (count == 0 && !waiting)
                SetStatus("No open rooms. Create one or refresh.");
        }

        private void Build()
        {
            GameObject card = CreateImage("RoomBrowserCard", transform, new Color(0.93f, 0.95f, 1f, 1f));
            RectTransform cardRect = card.GetComponent<RectTransform>();
            cardRect.anchorMin = cardRect.anchorMax = new Vector2(0.5f, 0.5f);
            cardRect.sizeDelta = new Vector2(720f, 520f);

            AddText(card.transform, "ONLINE ROOMS", new Vector2(0f, 225f), new Vector2(650f, 48f), 34f, FontStyles.Bold);
            statusText = AddText(card.transform, "Choose an input mode, then create or join a room.", new Vector2(0f, 180f), new Vector2(650f, 42f), 20f);

            inputModeButton = AddButton(card.transform, "InputMode", new Vector2(0f, 125f), new Vector2(360f, 46f), ToggleInputMode);
            inputModeText = inputModeButton.GetComponentInChildren<TMP_Text>();
            UpdateInputModeLabel();

            createButton = AddButton(card.transform, "CreateRoom", new Vector2(-185f, 68f), new Vector2(170f, 44f), () => createRequested?.Invoke());
            createButton.GetComponentInChildren<TMP_Text>().SetText("Create room");
            refreshButton = AddButton(card.transform, "RefreshRooms", new Vector2(0f, 68f), new Vector2(150f, 44f), () => refreshRequested?.Invoke());
            refreshButton.GetComponentInChildren<TMP_Text>().SetText("Refresh");
            closeButton = AddButton(card.transform, "CloseRooms", new Vector2(175f, 68f), new Vector2(150f, 44f), () => closeRequested?.Invoke());
            closeButtonText = closeButton.GetComponentInChildren<TMP_Text>();
            closeButtonText.SetText("Close");

            readyButton = AddButton(card.transform, "Ready", new Vector2(-95f, 68f), new Vector2(220f, 44f), () => readyRequested?.Invoke());
            readyButtonText = readyButton.GetComponentInChildren<TMP_Text>();
            readyButton.gameObject.SetActive(false);
            lobbyStatusText = AddText(card.transform, "Waiting for an opponent...", new Vector2(0f, -70f), new Vector2(650f, 150f), 23f);
            lobbyStatusText.gameObject.SetActive(false);

            roomsTitle = AddText(card.transform, "OPEN ROOMS", new Vector2(0f, 20f), new Vector2(640f, 30f), 20f, FontStyles.Bold);
            for (int index = 0; index < MaxVisibleRooms; index++)
            {
                Button room = AddButton(
                    card.transform,
                    $"Room_{index + 1}",
                    new Vector2(0f, -25f - index * 58f),
                    new Vector2(610f, 48f),
                    null);
                room.GetComponent<Image>().color = index % 2 == 0
                    ? new Color(0.72f, 0.79f, 0.97f, 1f)
                    : new Color(0.79f, 0.84f, 0.99f, 1f);
                roomButtons.Add(room);
                roomLabels.Add(room.GetComponentInChildren<TMP_Text>());
                room.gameObject.SetActive(false);
            }

            AddText(card.transform, "AI Script is available in offline modes only.", new Vector2(0f, -246f), new Vector2(620f, 24f), 15f);
        }

        private void ToggleInputMode()
        {
            selectedInputType = selectedInputType == InputType.UI
                ? InputType.LiveCommand
                : InputType.UI;
            UpdateInputModeLabel();
        }

        private void UpdateInputModeLabel()
        {
            if (inputModeText == null)
                return;

            inputModeText.SetText(selectedInputType == InputType.LiveCommand
                ? "Input: Live Commands"
                : "Input: Buttons / Keyboard");
        }

        private Button AddButton(Transform parent, string name, Vector2 position, Vector2 size, UnityEngine.Events.UnityAction action)
        {
            GameObject gameObject = CreateImage(name, parent, new Color(0.36f, 0.47f, 0.82f, 1f));
            RectTransform rect = gameObject.GetComponent<RectTransform>();
            rect.anchorMin = rect.anchorMax = new Vector2(0.5f, 0.5f);
            rect.anchoredPosition = position;
            rect.sizeDelta = size;

            Button button = gameObject.AddComponent<Button>();
            ColorBlock colors = button.colors;
            colors.highlightedColor = new Color(0.48f, 0.59f, 0.94f, 1f);
            colors.pressedColor = new Color(0.25f, 0.34f, 0.68f, 1f);
            colors.disabledColor = new Color(0.55f, 0.57f, 0.64f, 0.6f);
            button.colors = colors;
            if (action != null)
                button.onClick.AddListener(action);

            TMP_Text label = AddText(gameObject.transform, name, Vector2.zero, size, 19f, FontStyles.Bold);
            label.color = Color.white;
            return button;
        }

        private TMP_Text AddText(
            Transform parent,
            string value,
            Vector2 position,
            Vector2 size,
            float fontSize,
            FontStyles style = FontStyles.Normal)
        {
            var textObject = new GameObject("Text", typeof(RectTransform), typeof(CanvasRenderer), typeof(TextMeshProUGUI));
            textObject.transform.SetParent(parent, false);
            RectTransform rect = textObject.GetComponent<RectTransform>();
            rect.anchorMin = rect.anchorMax = new Vector2(0.5f, 0.5f);
            rect.anchoredPosition = position;
            rect.sizeDelta = size;

            TextMeshProUGUI text = textObject.GetComponent<TextMeshProUGUI>();
            text.font = font;
            text.fontSize = fontSize;
            text.fontStyle = style;
            text.alignment = TextAlignmentOptions.Center;
            text.color = new Color(0.12f, 0.14f, 0.22f, 1f);
            text.textWrappingMode = TextWrappingModes.Normal;
            text.richText = false;
            text.SetText(value);
            return text;
        }

        private static GameObject CreateImage(string name, Transform parent, Color color)
        {
            var gameObject = new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
            gameObject.transform.SetParent(parent, false);
            gameObject.GetComponent<Image>().color = color;
            return gameObject;
        }
    }
}
