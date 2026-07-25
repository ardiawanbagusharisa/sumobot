using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace SumoLeaderboard
{
    /// <summary>
    /// Drives the Leaderboards screen in MainMenu.unity. Binds to the hierarchy
    /// that already exists under MenuLeaderboards (rows, the filter dropdowns
    /// and the PlayerRank chip) and fills it with live data from
    /// LeaderboardService.
    ///
    /// Features: three filters (game mode / player mode / control mode) with
    /// PlayerPrefs persistence, tap-a-row stat details, and click-the-chip
    /// player rename.
    /// </summary>
    public class LeaderboardPanelController : MonoBehaviour
    {
        [Header("Optional explicit bindings (auto-resolved by name when null)")]
        public TMP_Dropdown GameModeDropdown; // cloned from PlayerModeDropdown at runtime when null
        public TMP_Dropdown PlayerModeDropdown;
        public TMP_Dropdown ControlModeDropdown;
        public RectTransform RowsContent;
        public GameObject PlayerRankChip;

        [Header("Row tier colors")]
        public Color GoldColor = new(0.961f, 0.773f, 0.259f, 1f);   // #F5C542
        public Color SilverColor = new(0.788f, 0.788f, 0.788f, 1f); // #C9C9C9
        public Color BronzeColor = new(0.780f, 0.482f, 0.247f, 1f); // #C77B3F

        private const string EmptyMessage = "No battles recorded yet - go fight!";
        private const string PrefsKeyGameMode = "Sumobot.Leaderboard.GameMode";
        private const string PrefsKeyPlayerMode = "Sumobot.Leaderboard.PlayerMode";
        private const string PrefsKeyControlMode = "Sumobot.Leaderboard.ControlMode";
        private const string LocalProfilePrefsKey = "Sumobot.Profile.Left";
        private const int MaxNameLength = 16;

        private readonly List<RowView> rows = new();
        private RowView templateRow;
        private Color defaultRowColor = Color.white;
        private TMP_Text rankChipRankText;
        private TMP_Text rankChipNameText;
        private TMP_InputField renameInput;
        private RectTransform statsPopup;
        private TMP_Text statsPopupText;
        private string expandedProfileId;
        private bool renaming;
        private bool initialized;

        private class RowView
        {
            public GameObject Root;
            public Image Background;
            public Button Button;
            public TMP_Text Rank;
            public TMP_Text PlayerName;
            public TMP_Text BotName;
            public TMP_Text Score;
            public GameObject Crown;
            public LeaderboardEntry Entry; // currently displayed entry (null for empty state)
        }

        #region Unity lifecycle

        void Awake()
        {
            EnsureInitialized();
        }

        void OnEnable()
        {
            EnsureInitialized();
            LeaderboardService.Instance.Updated += Refresh;
            expandedProfileId = null;
            Refresh();
        }

        void OnDisable()
        {
            if (LeaderboardService.HasInstance)
                LeaderboardService.Instance.Updated -= Refresh;
            CancelRename();
            CloseStatsPopup();
        }

        #endregion

        #region Setup

        private void EnsureInitialized()
        {
            if (initialized)
                return;
            initialized = true;

            Transform scroll = transform.Find("Scroll View");

            if (PlayerModeDropdown == null)
                PlayerModeDropdown = scroll.Find("DropdownPlayerModes").GetComponent<TMP_Dropdown>();
            if (ControlModeDropdown == null)
                ControlModeDropdown = scroll.Find("DropdownControlModes").GetComponent<TMP_Dropdown>();
            if (RowsContent == null)
                RowsContent = scroll.Find("Viewport/Content") as RectTransform;
            if (PlayerRankChip == null)
                PlayerRankChip = scroll.Find("PlayerRank").gameObject;

            rankChipRankText = PlayerRankChip.transform.Find("Text (TMP)").GetComponent<TMP_Text>();
            rankChipNameText = PlayerRankChip.transform.Find("Text (TMP) (1)").GetComponent<TMP_Text>();

            if (GameModeDropdown == null)
                GameModeDropdown = CreateGameModeDropdown();

            // The chip was copy-pasted with a stale persistent onClick
            // (Battle_Start); replace it with the rename action.
            Button chipButton = PlayerRankChip.GetComponent<Button>();
            if (chipButton != null)
            {
                chipButton.onClick = new Button.ButtonClickedEvent();
                chipButton.onClick.AddListener(BeginRename);
            }

            // Adopt the designer-placed rows as the initial pool.
            rows.Clear();
            foreach (Transform child in RowsContent)
            {
                RowView row = BindRow(child.gameObject);
                if (row != null)
                    rows.Add(row);
            }

            if (rows.Count > 0)
            {
                templateRow = rows[0];
                if (templateRow.Background != null)
                    defaultRowColor = templateRow.Background.color;
            }
            else
            {
                Logger.Error("[Leaderboard] No placeholder rows found under Scroll View/Viewport/Content.");
            }

            RestoreFilters();

            GameModeDropdown.onValueChanged.AddListener(OnFilterChanged);
            PlayerModeDropdown.onValueChanged.AddListener(OnFilterChanged);
            ControlModeDropdown.onValueChanged.AddListener(OnFilterChanged);
        }

        /// <summary>
        /// Builds the Game Mode filter by cloning the player-mode dropdown so it
        /// inherits the hand-drawn visual style, and places it one "slot" to the
        /// left of it (filter order: Game Mode | Player Mode | Control Mode).
        /// </summary>
        private TMP_Dropdown CreateGameModeDropdown()
        {
            GameObject clone = Instantiate(PlayerModeDropdown.gameObject, PlayerModeDropdown.transform.parent);
            clone.name = "DropdownGameModes";

            RectTransform playerRt = (RectTransform)PlayerModeDropdown.transform;
            RectTransform controlRt = (RectTransform)ControlModeDropdown.transform;
            RectTransform cloneRt = (RectTransform)clone.transform;

            // Same spacing as between the two existing dropdowns, mirrored left.
            float slotOffset = controlRt.anchoredPosition.x - playerRt.anchoredPosition.x;
            cloneRt.anchoredPosition = new Vector2(
                playerRt.anchoredPosition.x - slotOffset,
                playerRt.anchoredPosition.y);

            TMP_Dropdown dropdown = clone.GetComponent<TMP_Dropdown>();
            dropdown.ClearOptions();
            dropdown.AddOptions(new List<string> { "Multiplayer", "Campaign" });
            dropdown.value = (int)GameMode.Multiplayer;
            dropdown.RefreshShownValue();
            return dropdown;
        }

        private RowView BindRow(GameObject rowObject)
        {
            Transform t = rowObject.transform;
            Transform rank = t.Find("Text (TMP)");
            Transform name = t.Find("Text (TMP) (1)");
            Transform bot = t.Find("Text (TMP) (2)");
            Transform score = t.Find("Text (TMP) (3)");
            Transform crown = t.Find("Image");

            if (rank == null || name == null || bot == null || score == null)
                return null;

            RowView row = new()
            {
                Root = rowObject,
                Background = rowObject.GetComponent<Image>(),
                Button = rowObject.GetComponent<Button>(),
                Rank = rank.GetComponent<TMP_Text>(),
                PlayerName = name.GetComponent<TMP_Text>(),
                BotName = bot.GetComponent<TMP_Text>(),
                Score = score.GetComponent<TMP_Text>(),
                Crown = crown != null ? crown.gameObject : null,
            };

            if (row.Button != null)
            {
                // Rows carry a stale persistent onClick (Battle_Start) from
                // copy-paste; replace it with the detail toggle.
                row.Button.onClick = new Button.ButtonClickedEvent();
                row.Button.onClick.AddListener(() => OnRowClicked(row));
            }

            return row;
        }

        #endregion

        #region Filters

        private void RestoreFilters()
        {
            SetClamped(GameModeDropdown, PlayerPrefs.GetInt(PrefsKeyGameMode, 0));
            SetClamped(PlayerModeDropdown, PlayerPrefs.GetInt(PrefsKeyPlayerMode, 0));
            SetClamped(ControlModeDropdown, PlayerPrefs.GetInt(PrefsKeyControlMode, 0));
        }

        private static void SetClamped(TMP_Dropdown dropdown, int value)
        {
            dropdown.SetValueWithoutNotify(Mathf.Clamp(value, 0, dropdown.options.Count - 1));
            dropdown.RefreshShownValue();
        }

        private void SaveFilters()
        {
            PlayerPrefs.SetInt(PrefsKeyGameMode, GameModeDropdown.value);
            PlayerPrefs.SetInt(PrefsKeyPlayerMode, PlayerModeDropdown.value);
            PlayerPrefs.SetInt(PrefsKeyControlMode, ControlModeDropdown.value);
            PlayerPrefs.Save();
        }

        private void OnFilterChanged(int _)
        {
            if (SFXManager.Instance != null)
                SFXManager.Instance.Play2D("ui_accept");
            SaveFilters();
            expandedProfileId = null;
            Refresh();
        }

        #endregion

        #region Refresh

        public void Refresh()
        {
            if (!isActiveAndEnabled || rows.Count == 0)
                return;

            // Any data or filter change invalidates the open stat box.
            CloseStatsPopup();

            GameMode gameMode = (GameMode)GameModeDropdown.value;
            PlayerMode mode = (PlayerMode)PlayerModeDropdown.value;
            ControlCategory control = (ControlCategory)ControlModeDropdown.value;

            List<LeaderboardEntry> entries = LeaderboardService.Instance.GetTable(gameMode, mode, control);

            EnsureRowCapacity(entries.Count);

            if (entries.Count == 0)
            {
                ShowEmptyState();
            }
            else
            {
                for (int i = 0; i < rows.Count; i++)
                {
                    if (i < entries.Count)
                        FillRow(rows[i], i + 1, entries[i]);
                    else
                    {
                        rows[i].Entry = null;
                        rows[i].Root.SetActive(false);
                    }
                }
            }

            RefreshOwnRankChip(gameMode, mode, control);
        }

        private void EnsureRowCapacity(int needed)
        {
            while (rows.Count < needed)
            {
                GameObject clone = Instantiate(templateRow.Root, RowsContent);
                clone.name = $"Player ({rows.Count + 1})";
                RowView row = BindRow(clone);
                rows.Add(row);
            }
        }

        private void FillRow(RowView row, int rank, LeaderboardEntry entry)
        {
            row.Entry = entry;
            row.Root.SetActive(true);
            row.Rank.text = $"#{rank}";
            row.PlayerName.text = entry.PlayerName;
            row.BotName.text = entry.BotName;
            row.Score.text = entry.Rating.ToString();
            if (row.Crown != null)
                row.Crown.SetActive(true);

            if (row.Background != null)
            {
                row.Background.color = rank switch
                {
                    1 => GoldColor,
                    2 => SilverColor,
                    3 => BronzeColor,
                    _ => defaultRowColor,
                };
            }
        }

        /// <summary>Detail line shown inside the floating stats box.</summary>
        private static string StatsLine(LeaderboardEntry e)
        {
            int winRate = e.GamesPlayed > 0
                ? Mathf.RoundToInt(100f * e.Wins / e.GamesPlayed)
                : 0;
            return $"{e.Wins}W  {e.Losses}L  {e.Draws}D   |   {winRate}% win rate   |   best streak {e.BestStreak}";
        }

        private void OnRowClicked(RowView row)
        {
            if (row.Entry == null)
                return;

            if (SFXManager.Instance != null)
                SFXManager.Instance.Play2D("ui_accept");

            if (expandedProfileId == row.Entry.ProfileID)
            {
                CloseStatsPopup();
            }
            else
            {
                expandedProfileId = row.Entry.ProfileID;
                ShowStatsPopup(row);
            }
        }

        private void ShowEmptyState()
        {
            RowView row = rows[0];
            row.Entry = null;
            row.Root.SetActive(true);
            row.Rank.text = string.Empty;
            row.PlayerName.text = EmptyMessage;
            row.BotName.text = string.Empty;
            row.Score.text = string.Empty;
            if (row.Crown != null)
                row.Crown.SetActive(false);
            if (row.Background != null)
                row.Background.color = defaultRowColor;

            for (int i = 1; i < rows.Count; i++)
            {
                rows[i].Entry = null;
                rows[i].Root.SetActive(false);
            }
        }

        private void RefreshOwnRankChip(GameMode gameMode, PlayerMode mode, ControlCategory control)
        {
            if (renaming)
                return; // the chip is swapped for the rename input right now

            PlayerProfile local = GameManager.Instance.Left;
            if (local == null)
            {
                PlayerRankChip.SetActive(false);
                return;
            }

            // Always visible (it doubles as the rename button); "#-" when unranked.
            int rank = LeaderboardService.Instance.GetRank(local.ID, gameMode, mode, control);
            PlayerRankChip.SetActive(true);
            rankChipRankText.text = rank > 0 ? $"#{rank}" : "#-";
            rankChipNameText.text = local.Name;
        }

        #endregion

        #region Stats popup

        private void ShowStatsPopup(RowView row)
        {
            if (statsPopup == null)
                BuildStatsPopup();

            LeaderboardEntry e = row.Entry;
            string title = e.BotName != null && e.BotName != "-"
                ? $"<b>{e.PlayerName}</b>  ({e.BotName})"
                : $"<b>{e.PlayerName}</b>";
            statsPopupText.text = $"{title}\n{StatsLine(e)}";

            // Size the box snugly around the (unwrapped) two-line text.
            Vector2 textSize = statsPopupText.GetPreferredValues(statsPopupText.text, Mathf.Infinity, Mathf.Infinity);
            statsPopup.sizeDelta = textSize + new Vector2(40f, 18f);

            // Pin it under the clicked row (or above, when too close to the
            // bottom of the panel), horizontally centered on the row.
            RectTransform rowRt = (RectTransform)row.Root.transform;
            statsPopup.position = rowRt.position;

            float gap = 12f;
            float offset = rowRt.rect.height * 0.5f + statsPopup.sizeDelta.y * 0.5f + gap;
            Rect panelRect = ((RectTransform)transform).rect;
            bool fitsBelow = statsPopup.anchoredPosition.y - offset - statsPopup.sizeDelta.y * 0.5f > panelRect.yMin + 20f;

            statsPopup.anchoredPosition += new Vector2(0f, fitsBelow ? -offset : offset);

            statsPopup.SetAsLastSibling(); // render on top of everything in the panel
            statsPopup.gameObject.SetActive(true);
        }

        private void CloseStatsPopup()
        {
            expandedProfileId = null;
            if (statsPopup != null)
                statsPopup.gameObject.SetActive(false);
        }

        /// <summary>
        /// Builds the floating stat box once: a dark rounded panel (styled after
        /// the rank chip) with a centered two-line label, parented to the panel
        /// root so the scroll view's mask can't clip it.
        /// </summary>
        private void BuildStatsPopup()
        {
            GameObject go = new("StatsPopup", typeof(RectTransform), typeof(Image));
            statsPopup = (RectTransform)go.transform;
            statsPopup.SetParent(transform, false);
            statsPopup.anchorMin = statsPopup.anchorMax = new Vector2(0.5f, 0.5f);
            statsPopup.pivot = new Vector2(0.5f, 0.5f);

            Image bg = go.GetComponent<Image>();
            Image chipImage = PlayerRankChip.GetComponent<Image>();
            if (chipImage != null)
            {
                bg.sprite = chipImage.sprite;
                bg.type = chipImage.type;
                bg.color = chipImage.color; // match the dark chip style
            }
            else
            {
                bg.color = new Color(0.15f, 0.15f, 0.2f, 0.95f);
            }

            GameObject textGo = new("Text", typeof(RectTransform));
            RectTransform textRt = (RectTransform)textGo.transform;
            textRt.SetParent(statsPopup, false);
            textRt.anchorMin = Vector2.zero;
            textRt.anchorMax = Vector2.one;
            textRt.offsetMin = Vector2.zero;
            textRt.offsetMax = Vector2.zero;

            statsPopupText = textGo.AddComponent<TextMeshProUGUI>();
            statsPopupText.font = rankChipNameText.font;
            statsPopupText.fontSize = rankChipNameText.fontSize * 0.65f;
            statsPopupText.color = rankChipNameText.color;
            statsPopupText.alignment = TextAlignmentOptions.Center;
            statsPopupText.richText = true;
            statsPopupText.textWrappingMode = TextWrappingModes.NoWrap; // keep it exactly two lines

            // Clicking the box dismisses it.
            Button closeButton = go.AddComponent<Button>();
            closeButton.targetGraphic = bg;
            closeButton.onClick.AddListener(CloseStatsPopup);

            go.SetActive(false);
        }

        #endregion

        #region Rename

        private void BeginRename()
        {
            PlayerProfile local = GameManager.Instance.Left;
            if (local == null || renaming)
                return;

            if (renameInput == null)
                renameInput = BuildRenameInput();

            renaming = true;
            PlayerRankChip.SetActive(false);
            renameInput.gameObject.SetActive(true);
            renameInput.text = local.Name;
            renameInput.ActivateInputField();
        }

        private void ApplyRename(string value)
        {
            if (!renaming)
                return;

            string clean = SanitizeName(value);
            PlayerProfile local = GameManager.Instance.Left;

            if (local != null && !string.IsNullOrEmpty(clean) && clean != local.Name)
            {
                local.Rename(LocalProfilePrefsKey, clean);
                LeaderboardService.Instance.RenameProfile(local.ID, local.Name);
            }

            CancelRename();
            Refresh();
        }

        private void CancelRename()
        {
            renaming = false;
            if (renameInput != null)
                renameInput.gameObject.SetActive(false);
            if (PlayerRankChip != null)
                PlayerRankChip.SetActive(true);
        }

        /// <summary>Trim, strip TMP rich-text brackets, cap the length.</summary>
        private static string SanitizeName(string raw)
        {
            if (string.IsNullOrWhiteSpace(raw))
                return null;
            string clean = raw.Replace("<", string.Empty).Replace(">", string.Empty).Trim();
            if (clean.Length > MaxNameLength)
                clean = clean.Substring(0, MaxNameLength);
            return clean;
        }

        /// <summary>
        /// Builds an inline TMP_InputField styled after the rank chip and placed
        /// exactly over it, used to edit the local player name.
        /// </summary>
        private TMP_InputField BuildRenameInput()
        {
            RectTransform chipRt = (RectTransform)PlayerRankChip.transform;

            GameObject go = new("RenameInput", typeof(RectTransform), typeof(Image));
            RectTransform rt = (RectTransform)go.transform;
            rt.SetParent(chipRt.parent, false);
            rt.anchorMin = chipRt.anchorMin;
            rt.anchorMax = chipRt.anchorMax;
            rt.pivot = chipRt.pivot;
            rt.anchoredPosition = chipRt.anchoredPosition;
            rt.sizeDelta = chipRt.sizeDelta;

            Image bg = go.GetComponent<Image>();
            Image chipImage = PlayerRankChip.GetComponent<Image>();
            if (chipImage != null)
            {
                bg.sprite = chipImage.sprite;
                bg.type = chipImage.type;
            }
            bg.color = Color.white;

            GameObject area = new("Text Area", typeof(RectTransform), typeof(RectMask2D));
            RectTransform areaRt = (RectTransform)area.transform;
            areaRt.SetParent(rt, false);
            areaRt.anchorMin = Vector2.zero;
            areaRt.anchorMax = Vector2.one;
            areaRt.offsetMin = new Vector2(15f, 4f);
            areaRt.offsetMax = new Vector2(-15f, -4f);

            GameObject textGo = new("Text", typeof(RectTransform));
            RectTransform textRt = (RectTransform)textGo.transform;
            textRt.SetParent(areaRt, false);
            textRt.anchorMin = Vector2.zero;
            textRt.anchorMax = Vector2.one;
            textRt.offsetMin = Vector2.zero;
            textRt.offsetMax = Vector2.zero;

            TextMeshProUGUI text = textGo.AddComponent<TextMeshProUGUI>();
            text.font = rankChipNameText.font;
            text.fontSize = rankChipNameText.fontSize;
            text.color = Color.black;
            text.alignment = TextAlignmentOptions.MidlineLeft;

            TMP_InputField input = go.AddComponent<TMP_InputField>();
            input.targetGraphic = bg;
            input.textViewport = areaRt;
            input.textComponent = text;
            input.characterLimit = MaxNameLength;
            input.onEndEdit.AddListener(ApplyRename);

            go.SetActive(false);
            return input;
        }

        #endregion
    }
}
