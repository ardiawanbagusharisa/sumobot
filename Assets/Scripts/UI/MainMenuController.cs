using UnityEngine;
using System.Collections.Generic;

public class MainMenuController : MonoBehaviour
{
	public enum MenuState
	{
		MainMenu,
		GameModes,
		Campaign,
		Multiplayer,
		MultiplayerLocal,
		Leaderboards,
		Settings,
	}

	private MenuState currentState = MenuState.MainMenu;

	// Map each menu state to its GameObject
	[System.Serializable]
	public struct MenuMapping
	{
		public MenuState state;
		public GameObject menuObject;
	}

	public MenuMapping[] menus;

	public void Start() {
		SwitchMenu(MenuState.MainMenu);
	}

	// Call this to switch the menu
	public void SwitchMenu(MenuState newState) {
		currentState = newState;

		foreach (var menu in menus) {

			if (menu.menuObject != null)
				menu.menuObject.SetActive(menu.state == currentState);
		}
	}

	public void ShowMainMenu() => SwitchMenu(MenuState.MainMenu);
	public void ShowGameModes() => SwitchMenu(MenuState.GameModes);
	public void ShowCampaign() => SwitchMenu(MenuState.Campaign);
	public void ShowMultiplayer() => SwitchMenu(MenuState.Multiplayer);
	public void ShowMultiplayerLocal() => SwitchMenu(MenuState.MultiplayerLocal);
	public void ShowLeaderboards() => SwitchMenu(MenuState.Leaderboards);
	public void ShowSettings() => SwitchMenu(MenuState.Settings);

}