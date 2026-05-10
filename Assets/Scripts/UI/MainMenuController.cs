using UnityEngine;

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
		Market
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
		//SwitchMenu(MenuState.MainMenu);
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
	public void ShowMarket() => SwitchMenu(MenuState.Market);
	public void QuitGame() => Application.Quit();

	// Load Battle scene 
	public void GoToBattle(){
		SFXManager.Instance.Play2D("ui_accept"); 
		UnityEngine.SceneManagement.SceneManager.LoadScene("Battle");
	}

}