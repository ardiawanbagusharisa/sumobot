using SumoBot.Graph;
using SumoBot.Graph.Authoring;
using TMPro;
using UnityEngine;

// One editable parameter on a node (E3.2, Phase 4) — e.g. a Constant's value or an action's
// duration. It edits a numeric field, clamps to the ParamSpec's [Min, Max], and writes the value
// straight to the model (Document.SetParam). Built in code by NodeView, so no scene wiring.
public class ParamFieldView : MonoBehaviour
{
    private string nodeId;
    private string paramId;
    private float min;
    private float max;
    private GraphDocument document;
    private TMP_InputField input;

    public void Bind(string nodeId, ParamSpec spec, float current, GraphDocument doc,
                     TMP_InputField input, TMP_Text nameLabel)
    {
        this.nodeId = nodeId;
        paramId = spec.Id;
        min = spec.Min;
        max = spec.Max;
        document = doc;
        this.input = input;

        if (nameLabel != null) nameLabel.text = spec.DisplayName;
        input.contentType = TMP_InputField.ContentType.DecimalNumber;
        input.text = Format(Mathf.Clamp(current, min, max));
        input.onEndEdit.AddListener(OnEndEdit);
    }

    private void OnEndEdit(string raw)
    {
        float value = float.TryParse(raw, out float parsed) ? parsed : min;
        value = Mathf.Clamp(value, min, max);
        input.SetTextWithoutNotify(Format(value));
        document?.SetParam(nodeId, paramId, value);
    }

    private static string Format(float value) => value.ToString("0.###");
}
