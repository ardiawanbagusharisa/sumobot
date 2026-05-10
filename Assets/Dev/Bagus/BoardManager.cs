using UnityEngine;

public class BoardManager : MonoBehaviour
{
	public static BoardManager Instance;

	public GameObject connectionPrefab;
	public Transform connectionLayer;

	public GameObject unfoldInventory;
	public GameObject panelInventory;

	private PortUI pendingPort;
	private Connection tempConnection;

	void Awake() {
		Instance = this;
	}

	void Update() {
		if (Input.GetMouseButtonDown(1)) {
			if (tempConnection != null) {
				Destroy(tempConnection.gameObject);
				tempConnection = null;
				pendingPort = null;
			}
			return;
		}

		if (tempConnection != null && pendingPort != null) {
			Canvas canvas = connectionLayer.GetComponentInParent<Canvas>();
			// For Screen Space - Overlay canvases, camera should be null in utility functions
			Camera cam = (canvas != null && canvas.renderMode == RenderMode.ScreenSpaceOverlay) ? null : Camera.main;

			if (connectionLayer is RectTransform rectTransform) {
				if (RectTransformUtility.ScreenPointToWorldPointInRectangle(rectTransform, Input.mousePosition, cam, out Vector3 worldPoint)) {
					tempConnection.DrawToPoint(worldPoint);
				}
			} else {
				Vector3 mousePos = Input.mousePosition;
				mousePos.z = Mathf.Abs(Camera.main.transform.position.z - connectionLayer.position.z);
				if (mousePos.z == 0) mousePos.z = 10f;
				tempConnection.DrawToPoint(Camera.main.ScreenToWorldPoint(mousePos));
			}
		}
	}

	public void StartConnection(PortUI from) {
		if (tempConnection != null) {
			Destroy(tempConnection.gameObject);
		}

		pendingPort = from;

		GameObject go = Instantiate(connectionPrefab, connectionLayer);
		tempConnection = go.GetComponent<Connection>();
		tempConnection.from = from;
	}

	public void CompleteConnection(PortUI to) {
		if (pendingPort == null) return;

		// Prevent duplicate connections with the same from and to
		foreach (Connection conn in FindObjectsByType<Connection>(FindObjectsSortMode.None)) {
			if (conn != tempConnection && conn.from == pendingPort && conn.to == to) {
				Destroy(tempConnection.gameObject);
				pendingPort = null;
				tempConnection = null;
				return;
			}
		}

		tempConnection.to = to;
		pendingPort.connections.Add(tempConnection);
		to.connections.Add(tempConnection);
		tempConnection.RefreshLine();

		pendingPort = null;
		tempConnection = null;
	}

	public void ToggleInventory() {
		if (panelInventory != null) {
			SFXManager.Instance.Play2D("ui_accept");
			bool isActive = panelInventory.activeSelf;
			unfoldInventory.SetActive(isActive);
			panelInventory.SetActive(!isActive);
		}
	}

	public void BackToBotCreator() {
		SFXManager.Instance.Play2D("ui_accept");
		UnityEngine.SceneManagement.SceneManager.LoadScene("BotCreator");
	}

}