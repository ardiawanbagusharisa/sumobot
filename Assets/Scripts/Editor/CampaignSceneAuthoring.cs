using System;
using System.Linq;
using SumoCampaign;
using SumoInput;
using TMPro;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

public static class CampaignSceneAuthoring
{
    private static readonly string[] CampaignScenes =
    {
        "Assets/Scenes/Campaign_1_1.unity",
        "Assets/Scenes/Campaign_2_1.unity",
        "Assets/Scenes/Campaign_3_1.unity"
    };

    [MenuItem("Sumobot/Repair Campaign Scene UI")]
    public static void Repair()
    {
        foreach (string path in CampaignScenes)
        {
            var scene = EditorSceneManager.OpenScene(path);
            var view = scene.GetRootGameObjects()
                .SelectMany(root => root.GetComponentsInChildren<CampaignMissionView>(true))
                .Single();
            var canvas = scene.GetRootGameObjects().Single(root => root.name == "UI").transform;
            // Keep the mission result visible even if the battle state hides
            // Ongoing Battle during the five-second follow-through.
            view.Panel.transform.SetParent(canvas, false);
            view.Panel.transform.SetAsLastSibling();
            var sticker = canvas.Find("MissionCompleteSticker");
            if (sticker == null)
            {
                var panel = new GameObject("MissionCompleteSticker", typeof(RectTransform),
                    typeof(CanvasRenderer), typeof(Image));
                panel.transform.SetParent(canvas, false);
                sticker = panel.transform;
            }

            sticker.SetAsLastSibling();
            var stickerRect = (RectTransform)sticker;
            stickerRect.anchorMin = new Vector2(.5f, 1f);
            stickerRect.anchorMax = new Vector2(.5f, 1f);
            stickerRect.pivot = new Vector2(.5f, 1f);
            stickerRect.anchoredPosition = new Vector2(0f, -108f);
            stickerRect.sizeDelta = new Vector2(660f, 92f);
            var background = sticker.GetComponent<Image>();
            background.color = new Color(.12f, .52f, .37f, .95f);
            background.raycastTarget = false;

            var textTransform = sticker.Find("Label");
            if (textTransform == null)
            {
                var labelObject = new GameObject("Label", typeof(RectTransform),
                    typeof(CanvasRenderer), typeof(TextMeshProUGUI));
                labelObject.transform.SetParent(sticker, false);
                textTransform = labelObject.transform;
            }
            var textRect = (RectTransform)textTransform;
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.offsetMin = new Vector2(14f, 0f);
            textRect.offsetMax = new Vector2(-14f, 0f);
            var label = textTransform.GetComponent<TextMeshProUGUI>();
            label.font = view.Title.font;
            label.fontSize = 48f;
            label.fontStyle = FontStyles.Bold;
            label.alignment = TextAlignmentOptions.Center;
            label.color = Color.white;
            label.raycastTarget = false;
            label.text = "MISSION COMPLETE";
            view.CompletionSticker = sticker.gameObject;
            view.CompletionStickerLabel = label;
            sticker.gameObject.SetActive(false);
            EditorUtility.SetDirty(view);
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);
        }

        {
            var scene = EditorSceneManager.OpenScene("Assets/Scenes/MainMenu.unity");
            var menu = scene.GetRootGameObjects()
                .SelectMany(root => root.GetComponentsInChildren<Transform>(true))
                .Single(item => item.name == "MenuCampaign");
            var content = menu.GetComponentsInChildren<Transform>(true)
                .Single(item => item.Cast<Transform>()
                    .Count(child => child.name.StartsWith("ButtonCampaign", StringComparison.Ordinal)) == 15);
            foreach (Transform child in content)
                if (child.name.StartsWith("ButtonCampaign", StringComparison.Ordinal) &&
                    child.TryGetComponent(out Button button))
                {
                    button.interactable = true;
                    EditorUtility.SetDirty(button);
                }
            var rect = (RectTransform)content;
            rect.anchorMax = rect.anchorMin;
            var fitter = content.GetComponent<ContentSizeFitter>();
            fitter.horizontalFit = ContentSizeFitter.FitMode.PreferredSize;
            EditorUtility.SetDirty(fitter);
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);
        }

        Debug.Log("Campaign scene UI repaired and saved.");
    }

    [MenuItem("Sumobot/Verify Campaign Scene UI")]
    public static void Verify()
    {
        var levels = CampaignCatalog.Levels;
        Require(levels.Count == 15, "Expected 15 campaign definitions.");
        Require(levels.Select(level => level.Id).Distinct().Count() == 15,
            "Campaign level IDs are not unique.");
        foreach (var level in levels)
        {
            var input = level.Id[0] switch
            {
                '1' => InputType.UI,
                '2' => InputType.LiveCommand,
                '3' => InputType.Script,
                _ => throw new InvalidOperationException("Unexpected campaign level ID " + level.Id)
            };
            Require(level.Input == input, "Unexpected input mode for " + level.Id);
            Require(CampaignScenes.Any(scene => scene.EndsWith(level.Scene + ".unity",
                StringComparison.Ordinal)), "Missing scene for " + level.Id);
            Require(level.ArenaRadius == 8f || level.ArenaRadius == 10f || level.ArenaRadius == 12f,
                "Invalid radius for " + level.Id);
            Require(level.SpawnFraction < .8f && Mathf.Abs(level.OpponentSpawnFraction) < .8f,
                "Spawn outside safe arena for " + level.Id);
            if (input == InputType.Script)
                Require(level.Answers != null && level.Answers.Length == 4 &&
                    level.CorrectAnswer >= 0 && level.CorrectAnswer < 4,
                    "Invalid FSM question for " + level.Id);
        }

        foreach (string path in CampaignScenes)
        {
            var scene = EditorSceneManager.OpenScene(path);
            var view = scene.GetRootGameObjects()
                .SelectMany(root => root.GetComponentsInChildren<CampaignMissionView>(true))
                .Single();
            var canvas = scene.GetRootGameObjects().Single(root => root.name == "UI").transform;
            Require(view.Panel != null && view.Panel.transform.parent == canvas,
                "Mission result can be hidden by battle state in " + path);
            Require(view.CompletionSticker != null &&
                view.CompletionSticker.transform.parent == canvas &&
                view.CompletionStickerLabel != null &&
                !view.CompletionSticker.activeSelf,
                "Completion sticker is not authored in " + path);
            Require(view.StartButton != null && view.RetryButton != null &&
                view.MenuButton != null && view.AnswerButtons.Length == 4,
                "Mission controls are incomplete in " + path);
        }

        {
            var scene = EditorSceneManager.OpenScene("Assets/Scenes/MainMenu.unity");
            var menu = scene.GetRootGameObjects()
                .SelectMany(root => root.GetComponentsInChildren<Transform>(true))
                .Single(item => item.name == "MenuCampaign");
            var content = menu.GetComponentsInChildren<Transform>(true)
                .Single(item => item.Cast<Transform>()
                    .Count(child => child.name.StartsWith("ButtonCampaign", StringComparison.Ordinal)) == 15);
            var cards = content.Cast<Transform>()
                .Where(child => child.name.StartsWith("ButtonCampaign", StringComparison.Ordinal))
                .Select(child => child.GetComponent<Button>()).ToArray();
            Require(cards.All(card => card != null && card.interactable && card.gameObject.activeSelf),
                "One or more campaign cards are disabled.");
            var scroll = content.GetComponentInParent<ScrollRect>(true);
            Require(scroll != null && scroll.horizontal && scroll.horizontalScrollbar != null &&
                scroll.content == content, "Campaign cards cannot scroll horizontally.");
            Require(((RectTransform)content).anchorMin.x == ((RectTransform)content).anchorMax.x,
                "Campaign content has conflicting stretch and preferred width.");
        }
        Debug.Log("[CampaignVerifier] PASS: 15 definitions, three scene templates, sticker, and 15 accessible cards.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException("[CampaignVerifier] " + message);
    }
}
