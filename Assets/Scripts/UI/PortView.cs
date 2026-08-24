using SumoBot.Graph;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

// The visual for one port on a node (E3.2). It carries just enough identity to be a wire endpoint —
// which node, which port, direction and value type — and colours its dot by type. In Phase 3 it also
// owns the wiring gesture: drag off a port to pull a wire (WiringController draws it), drop on another
// port to connect, and right-click to cut every wire on this port. Built in code by NodeView, so the
// board needs no extra scene wiring.
public class PortView : MonoBehaviour,
    IBeginDragHandler, IDragHandler, IEndDragHandler, IDropHandler, IPointerClickHandler
{
    public string NodeId { get; private set; }
    public string PortId { get; private set; }
    public PortDirection Direction { get; private set; }
    public PortType Type { get; private set; }

    private WiringController wiring;

    /// <summary>World position of the dot — wires are drawn between these.</summary>
    public Vector3 WorldPosition => transform.position;

    /// <summary>Bind this port to a module's PortSpec: identity, dot colour, and label.</summary>
    public void Bind(string nodeId, PortSpec spec, Image dot, TMP_Text label)
    {
        NodeId = nodeId;
        PortId = spec.Id;
        Direction = spec.Direction;
        Type = spec.Type;

        if (dot != null) dot.color = spec.Type == PortType.Bool ? UITheme.Instance.PortBool : UITheme.Instance.PortNumber;
        if (label != null) label.text = spec.DisplayName;
    }

    private WiringController Wiring => wiring != null ? wiring : (wiring = GetComponentInParent<WiringController>());

    public void OnBeginDrag(PointerEventData eventData) => Wiring?.BeginWire(this);

    public void OnDrag(PointerEventData eventData) => Wiring?.UpdateWire(eventData.position);

    public void OnEndDrag(PointerEventData eventData) => Wiring?.CancelWire();

    public void OnDrop(PointerEventData eventData) => Wiring?.CompleteWire(this);

    public void OnPointerClick(PointerEventData eventData)
    {
        if (eventData.button == PointerEventData.InputButton.Right) Wiring?.RemovePort(this);
    }
}
