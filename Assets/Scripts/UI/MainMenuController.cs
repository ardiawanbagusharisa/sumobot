using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;

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
		Store, // renamed from Market — ordinal unchanged, MenuMapping[] in the scene serializes by int
		Login,
		Garage,
		Account,
		Workshop // keep last — MenuMapping serializes by int ordinal; appending avoids shifting existing entries
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
		// Returning from the node editor (E3.2) lands back on the Workshop panel; a normal boot
		// still starts at Login. The flag is consume-once so it never leaks into a later boot.
		SwitchMenu(WorkshopRouting.ConsumeReturnToWorkshop() ? MenuState.Workshop : MenuState.Login);
	}

	private void BuildCampaignButtons() {
		Transform campaignMenu = FindSceneObject("MenuCampaign");
		if (campaignMenu == null) return;

		Transform scrollView = FindDeep(campaignMenu, "Scroll View");
		Transform content = scrollView != null ? FindDeep(scrollView, "Content") : null;
		if (content == null) return;

		string[] levelIds = { "1-1", "2-1", "3-1" };
		string[] scenes = { "Campaign_1_1", "Campaign_2_1", "Campaign_3_1" };
		string[] labels = {
			"LEVEL 1-1  •  BUTTON CONTROLS",
			"LEVEL 2-1  •  LIVE COMMANDS",
			"LEVEL 3-1  •  FSM BOT"
		};

		List<Button> campaignButtons = new();
		foreach (Transform child in content) {
			if (child.name.StartsWith("ButtonCampaign") && child.TryGetComponent(out Button button))
				campaignButtons.Add(button);
		}

		for (int i = 0; i < levelIds.Length && i < campaignButtons.Count; i++) {
			Button button = campaignButtons[i];
			button.gameObject.name = $"ButtonCampaign {levelIds[i]}";
			TMP_Text label = button.GetComponentInChildren<TMP_Text>(true);
			if (label != null) label.text = labels[i];
			button.onClick = new Button.ButtonClickedEvent();
			string sceneName = scenes[i];
			button.onClick.AddListener(() => SceneManager.LoadScene(sceneName));
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
	public void ShowCampaign() => SwitchMenu(MenuState.Campaign);
	public void ShowMultiplayer() => SwitchMenu(MenuState.Multiplayer);
	public void ShowMultiplayerLocal() => SwitchMenu(MenuState.MultiplayerLocal);
	public void ShowLeaderboards() => SwitchMenu(MenuState.Leaderboards);
	public void ShowSettings() => SwitchMenu(MenuState.Settings);
	public void ShowStore() => SwitchMenu(MenuState.Store);
	public void ShowGarage() => SwitchMenu(MenuState.Garage);
	public void ShowAccount() => SwitchMenu(MenuState.Account);
	public void ShowWorkshop() => SwitchMenu(MenuState.Workshop);
	public void QuitGame() => Application.Quit();

	// Load Battle scene 
	public void GoToBattle(){
		SFXManager.Instance.Play2D("ui_accept"); 
		UnityEngine.SceneManagement.SceneManager.LoadScene("Battle");
	}

}
