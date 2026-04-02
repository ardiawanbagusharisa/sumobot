using UnityEngine;
using UnityEngine.EventSystems;
using System.Collections.Generic;

public class PortUI : MonoBehaviour, IPointerDownHandler
{
	public bool isOutput;
	public List<Connection> connections = new List<Connection>();

	public void Awake() {
		GetComponentInChildren<TMPro.TMP_Text>().text = isOutput ? "O" : "I";
	}

	public void OnPointerDown(PointerEventData eventData) {
		if (eventData.button == PointerEventData.InputButton.Right) {
			// Clear all connections connected to this port
			foreach (var conn in connections.ToArray()) { // copy array to avoid modifying collection while iterating
				if (conn != null) {
					conn.RemoveConnection();
				}
			}
			connections.Clear();
			return;
		}

		// Only respond to left clicks for starting or completing connections
		if (eventData.button != PointerEventData.InputButton.Left) return;

		Debug.Log("Port clicked: " + name);

		if (isOutput) {
			BoardManager.Instance.StartConnection(this);
		} else {
			BoardManager.Instance.CompleteConnection(this);
		}
	}

	public Vector3 GetWorldPosition() {
		return transform.position;
	}
}