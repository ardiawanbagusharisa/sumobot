using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;
using System.Linq;
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
	// Scene changes destroy this controller. Keep the last visible menu only
	// for this running game session, then consume it on the next menu load.
	private static MenuState? pendingReturnMenu;

	// Map each menu state to its GameObject
	[System.Serializable]
	public struct MenuMapping {
		public MenuState state;
		public GameObject menuObject;
	}

	public MenuMapping[] menus;

	public void Start() {
		CampaignCatalog.ClearSelection();
		if (GameServices.Auth?.IsSignedIn == true && pendingReturnMenu.HasValue)
			ShowReturnMenuOrMainMenu();
		else
			SwitchMenu(MenuState.Login);
	}

	private void OnDestroy() {
		if (currentState != MenuState.Login)
			pendingReturnMenu = currentState;
	}

	public void ShowReturnMenuOrMainMenu() {
		MenuState destination = pendingReturnMenu ?? MenuState.MainMenu;
		pendingReturnMenu = null;
		if (destination == MenuState.Campaign)
			ShowCampaign();
		else
			SwitchMenu(destination == MenuState.Login ? MenuState.MainMenu : destination);
	}

	private void BuildCampaignButtons() {
		Transform campaignMenu = FindSceneObject("MenuCampaign");
		if (campaignMenu == null) return;

		Transform content = campaignMenu.GetComponentsInChildren<Transform>(true)
			.FirstOrDefault(candidate => candidate.Cast<Transform>()
				.Any(child => child.name.StartsWith("ButtonCampaign")));
		if (content == null) return;

		List<Button> campaignButtons = new();
		foreach (Transform child in content) {
			if (child.name.StartsWith("ButtonCampaign") && child.TryGetComponent(out Button button))
				campaignButtons.Add(button);
		}

		if (campaignButtons.Count == 0) return;
		if (campaignButtons.Count != CampaignCatalog.Levels.Count) {
			Debug.LogError($"Expected {CampaignCatalog.Levels.Count} authored campaign buttons, found {campaignButtons.Count}.");
			return;
		}

		for (int i = 0; i < CampaignCatalog.Levels.Count; i++) {
			CampaignLevelDefinition level = CampaignCatalog.Levels[i];
			Button button = campaignButtons[i];
			button.interactable = true;
			button.gameObject.name = $"ButtonCampaign {level.Id}";
			TMP_Text label = button.GetComponentInChildren<TMP_Text>(true);
			bool completed = GameServices.PlayerData?.Current?.CompletedCampaignLevels?.Contains(level.Id) ?? false;
			if (label != null)
				label.text = $"Lv {level.Id} {(completed ? "(Cleared)" : "")} \n{level.Title}"; //level.Title.ToUpperInvariant()
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
