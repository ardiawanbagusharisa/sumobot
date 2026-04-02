using UnityEditor.MemoryProfiler;
using UnityEngine;
using UnityEngine.EventSystems;
using System.Collections.Generic;

public class PortUI : MonoBehaviour, IPointerDownHandler
{
	public bool isOutput;
	public List<Connection> connections = new List<Connection>();

	public void OnPointerDown(PointerEventData eventData) {
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