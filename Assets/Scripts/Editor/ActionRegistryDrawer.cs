#if UNITY_EDITOR
namespace SumoEditor
{
    using SumoHelper;
    using UnityEditor;
    using UnityEngine;


    [CustomPropertyDrawer(typeof(EventRegistry))]
    public class ActionRegistryDrawer : PropertyDrawer
    {
        private bool _foldout = true;

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            // Draw the label with a foldout
            _foldout = EditorGUI.Foldout(
                new Rect(position.x, position.y, position.width, EditorGUIUtility.singleLineHeight),
                _foldout,
                label,
                true
            );

            if (!_foldout)
                return;

            EditorGUI.indentLevel++;

            // Use reflection to get access to entries
            var target = property.serializedObject.targetObject;
            var field = fieldInfo.GetValue(target) as EventRegistry;

            if (field == null)
            {
                EditorGUI.LabelField(position, "Unable to load ActionRegistry");
                return;
            }

            float y = position.y + EditorGUIUtility.singleLineHeight + 4;

            foreach (var entry in field.Events)
            {
                var key = entry.Key;
                var action = entry.Value;

                var headerRect = new Rect(position.x, y, position.width, EditorGUIUtility.singleLineHeight);
                EditorGUI.LabelField(headerRect, $"{key} - {action.SubscribersCount} subscriber(s)", EditorStyles.boldLabel);
                y += EditorGUIUtility.singleLineHeight;

                foreach (var subscriber in action.GetSubscriberDescriptions())
                {
                    var subRect = new Rect(position.x + 20, y, position.width - 20, EditorGUIUtility.singleLineHeight);
                    EditorGUI.LabelField(subRect, $"↳ {subscriber}");
                    y += EditorGUIUtility.singleLineHeight;
                }

                y += 4;
            }

            EditorGUI.indentLevel--;
        }

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            var target = property.serializedObject.targetObject;
            var field = fieldInfo.GetValue(target) as EventRegistry;

            if (!_foldout || field == null)
                return EditorGUIUtility.singleLineHeight;

            int lines = 1;
            foreach (var entry in field.Events)
            {
                lines += 1; // header
                lines += Mathf.Max(1, entry.Value.SubscribersCount); // subscriber lines
                lines += 1; // spacing
            }

            return lines * (EditorGUIUtility.singleLineHeight + 2);
        }
    }
}
#endif
