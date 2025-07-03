using UnityEngine;
using TMPro;
using UnityEngine.UI;
using System;
using System.Collections.Generic;
using System.Collections;
using System.Linq;
using SumoCore;
using SumoBot;

namespace SumoInput
{
    public class CommandSystem : MonoBehaviour
    {
        #region UI Elements properties 
        [Header("UI Elements")]
        public TMP_InputField inputField;
        public TextMeshProUGUI display;
        public ScrollRect scrollDisplay;
        public Button foldingButton;
        #endregion

        #region Command properties
        private Dictionary<string, Action<string>> allCommands;
        private string defaultText = CommandMessage.Default;
        private string helperTexts = CommandMessage.Help;
        #endregion

        #region Runtime properties 
        private bool isUnfolded = false;
        private InputProvider inputProvider;
        private SumoAPI api;
        #endregion

        #region Unity methods

        void OnEnable()
        {
            allCommands = new Dictionary<string, Action<string>>()
                {
                    // [Todo] Consider to put "()" for correct format.
                    { "accelerate", CommandAccelerate },
                    { "turnleft", CommandTurnLeft },
                    { "turnright", CommandTurnRight },
                    { "dash", (s) => Dash() },
                    { "skill", (s) => Skill() },
                    { "clear", (s) => ClearTerminal() },
                    { "help", (s) => AddMessageToDisplay(helperTexts) },
                    { "open", (s) => OpenTerminal() },
                    { "close", (s) => CloseTerminal() },
                    { "getstatus", (s) => GetStatus() },   //[Todo] Implement get log from log manager. 
                };

            inputField.onValueChanged.AddListener(OnTyping);
            inputField.onSubmit.AddListener(OnSubmit);

            DisplayMessage(defaultText);
        }

        private void Update()
        {
            CheckPlayerInput();
        }
        #endregion

        #region Command system methods
        public void InitCommandSystem(SumoAPI api)
        {
            //[Todo] Handle null later. 
            inputProvider = GetComponent<InputProvider>();
            this.api = api;
        }

        private void CheckPlayerInput()
        {
            if (Input.GetKeyDown(KeyCode.Tab) && !string.IsNullOrEmpty(inputField.text))
            {
                string currentText = inputField.text;
                string suggestion = GetClosestCommand(currentText);

                // [Todo] Move as autocomplete function. 
                if (!string.IsNullOrEmpty(suggestion))
                {
                    string addParens = (suggestion != "help" && suggestion != "open" && suggestion != "close") ? "()" : "";

                    inputField.text = suggestion + addParens;
                    inputField.caretPosition = inputField.text.Length - (inputField.text.Contains("(") ? 1 : 0);
                }
            }
        }

        private void DisplayMessage(string message)
        {
            ClearTerminal();
            display.text = message;
            StartCoroutine(ScrollDisplayToBottom());
        }

        private void AddMessageToDisplay(string message)
        {
            display.text += "\n" + message;
            StartCoroutine(ScrollDisplayToBottom());
        }

        private string GetLastMessage()
        {
            int lastIndex = display.text.LastIndexOf('\n');
            return lastIndex == -1 ? display.text : display.text.Substring(lastIndex + 1);
        }

        private void RemoveLastSuggestion()
        {
            int lastIndex = display.text.LastIndexOf('\n');
            if (lastIndex >= 0)
            {
                display.text = display.text.Substring(0, lastIndex);
            }
        }

        private IEnumerator ScrollDisplayToBottom()
        {
            yield return null;

            LayoutRebuilder.ForceRebuildLayoutImmediate((RectTransform)display.transform);
            scrollDisplay.verticalNormalizedPosition = 0f; // 0 = bottom, 1 = top.
        }

        private void ClearTerminal()
        {
            display.text = "";
            scrollDisplay.verticalNormalizedPosition = 1f;
            LayoutRebuilder.ForceRebuildLayoutImmediate((RectTransform)display.transform);
        }

        private void OpenTerminal()
        {
            if (!isUnfolded)
                ToggleFold();
        }

        private void CloseTerminal()
        {
            if (isUnfolded)
                ToggleFold();
        }

        private void OnTyping(string currentInput)
        {
            string closest = GetClosestCommand(currentInput);
            string suggestion = CommandMessage.SuggestedCmd;

            if (!string.IsNullOrEmpty(closest))
            {
                // Add suggested command if it's not already in the last message. Becareful, becase there is a word "help" in the text. 
                string lastMessage = GetLastMessage();

                if (lastMessage.Contains("help") || !lastMessage.Contains(closest))
                {
                    if (lastMessage.Contains(suggestion))
                        RemoveLastSuggestion();

                    AddMessageToDisplay("> " + suggestion + ": " + closest);
                }
            }
            else if (string.IsNullOrEmpty(currentInput) && GetLastMessage().Contains(suggestion))
            {
                RemoveLastSuggestion();
            }
        }

        private void OnSubmit(string input)
        {
            OpenTerminal();

            if (!string.IsNullOrEmpty(input))
            {
                AddMessageToDisplay("> " + CommandMessage.TryExecute + " " + input + ".");
                TryExecute(input);
                inputField.text = "";
                inputField.ActivateInputField();
            }
        }

        private string GetClosestCommand(string input)
        {
            // [Todo] Check if command has "()". Ex: help
            if (!string.IsNullOrEmpty(input))
            {
                foreach (var cmd in allCommands.Keys)
                {
                    if (cmd.StartsWith(input, StringComparison.OrdinalIgnoreCase))
                    {
                        return cmd;
                    }
                }
            }
            return "";
        }

        public void ToggleFold()
        {
            isUnfolded = !isUnfolded;
            scrollDisplay.gameObject.SetActive(isUnfolded);

            var foldButtonrect = foldingButton.GetComponent<RectTransform>();
            var scale = foldButtonrect.localScale;
            scale.y = isUnfolded ? -Mathf.Abs(scale.y) : Mathf.Abs(scale.y);
            foldButtonrect.localScale = scale;
        }
        #endregion

        #region Commands Execution 
        private void TryExecute(string input)
        {
            string trimmed = input.Trim();
            if (string.IsNullOrEmpty(trimmed))
            {
                AddMessageToDisplay($"> {CommandMessage.EmptyCmd}\n");
                return;
            }

            // Extract command name and arguments 
            int parenStart = trimmed.IndexOf('(');
            int parenEnd = trimmed.IndexOf(')');
            string command = "";
            string rawArgs = "";

            if (parenStart > 0 && parenEnd > parenStart)
            {
                command = trimmed.Substring(0, parenStart);
                rawArgs = trimmed.Substring(parenStart + 1, parenEnd - parenStart - 1);
            }
            else
            {
                command = trimmed;
            }

            // Validate if command exists
            if (!allCommands.TryGetValue(command, out var action))
            {
                AddMessageToDisplay($"> {CommandMessage.UnknownCmd} \"{command}\".");
                return;
            }

            // Argument validation per command
            bool hasParens = parenStart != -1 && parenEnd != -1;
            switch (command)
            {
                case "accelerate":
                case "turnleft":
                case "turnright":
                    if (!hasParens)
                    {
                        AddMessageToDisplay($"> {command}{CommandMessage.InvalidParensArgs}.");
                        return;
                    }

                    if (rawArgs.Count() == 0)
                    {
                        AddMessageToDisplay($"> {CommandMessage.InvalidArgsExpect} \"{command}{CommandMessage.InvalidArgs}\".");
                        return;
                    }

                    try
                    {
                        action.Invoke(rawArgs);
                    }
                    catch (Exception ex)
                    {
                        AddMessageToDisplay("> Invalid Arguments: " + ex.Message);
                    }
                    break;

                case "dash":
                case "skill":
                case "clear":
                case "getstatus":
                    if (!hasParens)
                    {
                        AddMessageToDisplay($"> {command}() {CommandMessage.InvalidParens}.");
                        return;
                    }

                    if (!string.IsNullOrWhiteSpace(rawArgs))
                    {
                        AddMessageToDisplay($"> {command}() {CommandMessage.InvalidNoArgs}");
                        return;
                    }

                    action.Invoke("");
                    break;

                case "open":
                case "close":
                case "help":
                    if (hasParens)
                    {
                        AddMessageToDisplay($"> {command} {CommandMessage.InvalidNoParens}.");
                        return;
                    }

                    action.Invoke("");
                    break;

                default:
                    AddMessageToDisplay($"> {CommandMessage.InvalidDefault} {command}.");
                    break;
            }

            if (command == "clear")
            {
                int firstIndex = display.text.IndexOf('\n');
                if (firstIndex >= 0)
                    display.text = display.text.Substring(firstIndex + 1);

                DisplayMessage(defaultText);
            }
            else if (!string.IsNullOrEmpty(rawArgs))
                AddMessageToDisplay($"> {command} {CommandMessage.ExecutedWithArgs} \"{rawArgs}\".");
            else
                AddMessageToDisplay($"> {command} {CommandMessage.Executed}.");
        }

        private void CommandAccelerate(string arg)
        {
            inputProvider.EnqueueCommand(new AccelerateAction(InputType.LiveCommand, float.Parse(arg)));
            AddMessageToDisplay("> Executing accelerate(" + arg + ").");

        }

        private void CommandTurnLeft(string arg)
        {
            ISumoAction turnLeft = new TurnAction(InputType.LiveCommand, ActionType.TurnLeftWithAngle, float.Parse(arg));
            inputProvider.EnqueueCommand(turnLeft);
            AddMessageToDisplay("> Executing TurnLeft(" + arg + ").");
        }

        private void CommandTurnRight(string arg)
        {
            ISumoAction turnRight = new TurnAction(InputType.LiveCommand, ActionType.TurnRightWithAngle, float.Parse(arg));
            inputProvider.EnqueueCommand(turnRight);
            AddMessageToDisplay("> Executing TurnRight(" + arg + ").");
        }

        private void Dash()
        {
            inputProvider.EnqueueCommand(new DashAction(InputType.LiveCommand));
            AddMessageToDisplay("> Executing Dash().");
        }

        private void Skill()
        {
            inputProvider.EnqueueCommand(new SkillAction(InputType.LiveCommand));
            AddMessageToDisplay("> Executing Skill().");
        }

        private void GetStatus()
        {
            string status = $"> Executing GetStatus.\n\n {api}";

            AddMessageToDisplay(status);
        }
        #endregion

        #region Inner Class CommandMessage 
        public class CommandMessage
        {
            public static string Default = "> Enter command, type \"help\" to show all commands, and press \"tab\" for autocomplete.";
            public static string Help = "> Available Commands:\n" +
                                            "   * accelerate(d)" +
                                                " -> Accelerate the robot forward within duration \"d\" (second).\n" +
                                            "   * turnleft(a)" +
                                                " -> Turn the robot left with given angle \"a\" (0-180 degree).\n" +
                                            "   * turnright(a)" +
                                                " -> Turn the robot right with given angle \"a\" (0-180 degree).\n" +
                                            "   * dash()" +
                                                " -> Dash the robot forward with x power.\n" +
                                            "   * skill()" +
                                                " -> Activate the robot's special skill.\n" +
                                            "   * clear()" +
                                                " -> ClearTerminal the command terminal.\n" +
                                            "   * open" +
                                                " -> Open the command terminal.\n" +
                                            "   * close" +
                                                " -> Close the command terminal.\n" +
                                            "   * help" +
                                                " -> Show all available commands.";

            public static string SuggestedCmd = "Suggested command";
            public static string TryExecute = "Try executing";
            public static string EmptyCmd = "Empty command";
            public static string UnknownCmd = "Unknown command!";

            public static string InvalidParensArgs = "(x) must include parentheses";
            public static string InvalidParens = "must include parentheses";
            public static string InvalidArgsExpect = "Invalid argument. Expected float for";
            public static string InvalidArgs = "(x)";
            public static string InvalidNoArgs = "does not take any arguments";
            public static string InvalidNoParens = "should not have parentheses";
            public static string InvalidDefault = "Unhandled command:";

            public static string ExecutedWithArgs = "executed with argument:";
            public static string Executed = "is executed";

        }
        #endregion

    }
}