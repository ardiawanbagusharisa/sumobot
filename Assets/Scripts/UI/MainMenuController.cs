using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;
using SumoCampaign;
using SumoServices;

public class MainMenuController : MonoBehaviour
{
	public enum MenuState {
		MainMenu,
		GameModes,
		Campaign,
		Multiplayer,
		MultiplayerLocal,
		Leaderboards,
		Settings,
		Market,
		Login
	}

	private MenuState currentState = MenuState.MainMenu;

	// Map each menu state to its GameObject
	[System.Serializable]
	public struct MenuMapping {
		public MenuState state;
		public GameObject menuObject;
	}

	public MenuMapping[] menus;

	public void Start() {
		CampaignCatalog.ClearSelection();
		SwitchMenu(MenuState.Login);
	}

	private void BuildCampaignButtons() {
		Transform campaignMenu = FindSceneObject("MenuCampaign");
		if (campaignMenu == null) return;

		Transform scrollView = FindDeep(campaignMenu, "Scroll View");
		Transform content = scrollView != null ? FindDeep(scrollView, "Content") : null;
		if (content == null) return;

		List<Button> campaignButtons = new();
		foreach (Transform child in content) {
			if (child.name.StartsWith("ButtonCampaign") && child.TryGetComponent(out Button button))
				campaignButtons.Add(button);
		}

		if (campaignButtons.Count == 0) return;
		while (campaignButtons.Count < CampaignCatalog.Levels.Count) {
			GameObject card = Instantiate(campaignButtons[campaignButtons.Count - 1].gameObject, content);
			campaignButtons.Add(card.GetComponent<Button>());
		}

		for (int i = 0; i < CampaignCatalog.Levels.Count; i++) {
			CampaignCatalog.Level level = CampaignCatalog.Levels[i];
			Button button = campaignButtons[i];
			button.gameObject.name = $"ButtonCampaign {level.Id}";
			TMP_Text label = button.GetComponentInChildren<TMP_Text>(true);
			bool completed = GameServices.PlayerData?.Current?.CompletedCampaignLevels?.Contains(level.Id) ?? false;
			if (label != null)
				label.text = $"LEVEL {level.Id}  •  {level.Title.ToUpperInvariant()}" +
					(completed ? "  ✓" : "");
			button.onClick = new Button.ButtonClickedEvent();
			button.onClick.AddListener(() => CampaignCatalog.Launch(level));
		}
	}

	private static Transform FindDeep(Transform parent, string objectName) {
		if (parent.name == objectName) return parent;
		foreach (Transform child in parent) {
			Transform found = FindDeep(child, objectName);
			if (found != null) return found;
		}
		return null;
	}

	private static Transform FindSceneObject(string objectName) {
		foreach (Transform candidate in Resources.FindObjectsOfTypeAll<Transform>()) {
			if (candidate.name == objectName && candidate.gameObject.scene.IsValid())
				return candidate;
		}
		return null;
	}

	// Call this to switch the menu
	public void SwitchMenu(MenuState newState) {
		currentState = newState;

		foreach (var menu in menus) {

			if (menu.menuObject != null)
				menu.menuObject.SetActive(menu.state == currentState);
		}

		SFXManager.Instance.Play2D("ui_accept");
	}

	public void ShowMainMenu() => SwitchMenu(MenuState.MainMenu);
	public void ShowGameModes() => SwitchMenu(MenuState.GameModes);
	public void ShowCampaign() {
		SwitchMenu(MenuState.Campaign);
		BuildCampaignButtons();
	}
	public void ShowMultiplayer() => SwitchMenu(MenuState.Multiplayer);
	public void ShowMultiplayerLocal() => SwitchMenu(MenuState.MultiplayerLocal);
	public void ShowLeaderboards() => SwitchMenu(MenuState.Leaderboards);
	public void ShowSettings() => SwitchMenu(MenuState.Settings);
	public void ShowMarket() => SwitchMenu(MenuState.Market);
	public void QuitGame() => Application.Quit();

	// Load Battle scene 
	public void GoToBattle(){
		SFXManager.Instance.Play2D("ui_accept"); 
		UnityEngine.SceneManagement.SceneManager.LoadScene("Battle");
	}

}
