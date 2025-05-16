using UnityEngine;
using TMPro;
using UnityEngine.UI;
using System;
using System.Collections.Generic;
using System.Collections;
using System.Security.Cryptography.X509Certificates;
using CoreSumoRobot;

public class CommandSystem : MonoBehaviour
{
    #region UI variables 
    [Header("UI Elements")]
    public TMP_InputField inputField;
    public TextMeshProUGUI display;
    public ScrollRect scrollDisplay;
    public Button foldingButton;
    #endregion

    #region Command variables
    private Dictionary<string, Action<string>> allCommands;
    private string defaultText = "> Enter command, type \"help\" to show all commands, and press \"tab\" for autocomplete.";
    private string helperTexts =
        "> Available Commands:\n" +
        "- accelerate(d)" +
            " -> Accelerate the robot forward within duration \"d\" (second).\n" +
        "- turnleft(a)" +
            " -> Turn the robot left with given angle \"a\" (degree).\n" +
        "- turnright(a)" +
            " -> Turn the robot right with given angle \"a\" (degree).\n" +
        "- dash()" +
            " -> Dash the robot forward with x power.\n" +
        "- skill()" +
            " -> Activate the robot's special skill.\n" +
        "- clear()" +
            " -> Clear the command terminal.\n" +
        "- help" +
            " -> Show all available commands.\n";
    #endregion

    #region Folding Button
    public bool isUnfolded = true;
    #endregion

   private InputProvider inputProvider;

    private void Start()
    {
        //[Todo] consider to move to onenable()
        initCommandSystem();
    }

    private void Update() 
    {
        CheckPlayerInput();
    }

    private void initCommandSystem()
    {
        //[Todo] Handle null later. 
        inputProvider = GetComponent<InputProvider>();

        allCommands = new Dictionary<string, Action<string>>()
        {
            // [Todo] Consider to put "()" for correct format.
            { "accelerate", CommandAccelerate },
            { "turnleft", CommandTurnLeft },
            { "turnright", CommandTurnRight },
            { "dash", (s) => Dash() },
            { "skill", (s) => AddMessageToDisplay("Executing skill()") },
            { "getstatus", (s) => AddMessageToDisplay("Executing getstatus()") },
            { "clear", (s) => Clear() },
            { "help", (s) => AddMessageToDisplay(helperTexts) },
        };


        inputField.onValueChanged.AddListener(OnTyping);
        inputField.onSubmit.AddListener(OnSubmit);

        DisplayMessage(defaultText);

        isUnfolded = scrollDisplay.gameObject.activeSelf;
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
                inputField.text = suggestion + (suggestion == "help" ? "" : "()");
                inputField.caretPosition = inputField.text.Length - (inputField.text.Contains("(") ? 1 : 0);
            }
        }
    }

    private void DisplayMessage(string message)
    {
        Clear();
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

    private IEnumerator ScrollDisplayToBottom() {
        yield return null;

        LayoutRebuilder.ForceRebuildLayoutImmediate((RectTransform)display.transform);
        scrollDisplay.verticalNormalizedPosition = 0f; // 0 = bottom, 1 = top.
    }

    private void Clear()
    {
        display.text = "";
        scrollDisplay.verticalNormalizedPosition = 1f;
        LayoutRebuilder.ForceRebuildLayoutImmediate((RectTransform)display.transform);
    }

    private void OnTyping(string currentInput)
    {
        string closest = GetClosestCommand(currentInput);
        if (!string.IsNullOrEmpty(closest))
        {
            // Add suggested command if it's not already in the last message. 
            string lastMessage = GetLastMessage();
            Debug.Log("Closest1: " + closest);
            if (lastMessage.Contains("help") || !lastMessage.Contains(closest))
            {
                Debug.Log("Closest2: " + closest);
                if (lastMessage.Contains("Suggested Command")) 
                {
                    RemoveLastSuggestion();
                }
                AddMessageToDisplay("> Suggested Command: " + closest);
            }
        }
        // Handle if the input is empty and the last message is a suggestion.
        else if (string.IsNullOrEmpty(currentInput) && GetLastMessage().Contains("Suggested Command"))
        {
            RemoveLastSuggestion();
        }
    }

    private void OnSubmit(string input)
    {
        ToggleFold();

        if (!string.IsNullOrEmpty(input)) 
        {
            AddMessageToDisplay("> Try executing " + input + ".");
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

    #region Commands Execution 
    private void TryExecute(string input)
    {
        string trimmed = input.Trim();
        if (string.IsNullOrEmpty(trimmed))
        {
            AddMessageToDisplay("> Empty command.\n");
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

        // Validate if the command exists
        if (!allCommands.TryGetValue(command, out var action))
        {
            AddMessageToDisplay($"> Unknown command: {command}.");
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
                    AddMessageToDisplay($"> {command}(x) must include parentheses.");
                    return;
                }

                if (!float.TryParse(rawArgs, out float floatVal))
                {
                    AddMessageToDisplay($"> Invalid argument. Expected float for \"{command}(x)\".");
                    return;
                }

                action.Invoke(rawArgs);
                break;

            case "dash":
            case "skill":
            case "clear":
            case "getstatus":
                if (!hasParens)
                {
                    AddMessageToDisplay($"> {command}() must include parentheses.");
                    return;
                }

                if (!string.IsNullOrWhiteSpace(rawArgs))
                {
                    AddMessageToDisplay($"> {command}() does not take any arguments.");
                    return;
                }

                action.Invoke("");
                break;

            case "help":
                if (hasParens)
                {
                    AddMessageToDisplay("> 'help' should not have parentheses.");
                    return;
                }

                action.Invoke("");
                break;

            default:
                AddMessageToDisplay($"> Unhandled command: {command}");
                break;
        }

        if (!string.IsNullOrEmpty(rawArgs))
        {
            AddMessageToDisplay($"> {command} executed with argument: \"{rawArgs}\".");
        }
        else {
            AddMessageToDisplay($"> {command} executed.");
            if (command.Contains("clear"))
            {
                int firstIndex = display.text.IndexOf('\n');
                display.text = display.text.Substring(firstIndex + 1);
                AddMessageToDisplay(defaultText);
            }
        }
        
    }


    private void CommandAccelerate(string arg)
    {
       inputProvider.EnqueueCommand(new AccelerateTimeAction(float.Parse(arg)));
        AddMessageToDisplay("> Executing accelerate(" + arg + ").\n");
    }

    private void CommandTurnLeft(string arg)
    {
        inputProvider.EnqueueCommand(new TurnLeftAngleAction(float.Parse(arg)));
        AddMessageToDisplay("> Executing TurnLeft(" + arg + ").\n");
    }

    private void CommandTurnRight(string arg)
    {
        inputProvider.EnqueueCommand(new TurnRightAngleAction(float.Parse(arg)));
        AddMessageToDisplay("> Executing TurnRight(" + arg + ").\n");
    }

    private void Dash()
    {
        inputProvider.EnqueueCommand(new DashAction(InputType.LiveCommand));
        AddMessageToDisplay("> Executing Dash().\n");
    }
    #endregion

    #region Folding Button 
    public void ToggleFold()
    {
        Debug.Log("ToggleFold: "+isUnfolded);
        isUnfolded = !isUnfolded;
        scrollDisplay.gameObject.SetActive(isUnfolded);
        Debug.Log("ToggleFold new: " + isUnfolded);

        var foldButtonrect = foldingButton.GetComponent<RectTransform>();
        var scale = foldButtonrect.localScale;
        scale.y = isUnfolded ? Mathf.Abs(scale.y) : -Mathf.Abs(scale.y);
        foldButtonrect.localScale = scale;
    }
    #endregion

    #region Inner Class CommandMessage 
    public class CommandMessage 
    { 
        public static string DefaultText = "> Enter command, type \"help\" to show all commands, and press \"tab\" for autocomplete.";
        public static string HelpText = "> Available Commands:\n" +
                                        "- accelerate(d)" +
                                            " -> Accelerate the robot forward within duration \"d\" (second).\n" +
                                        "- turnleft(a)" +
                                            " -> Turn the robot left with given angle \"a\" (degree).\n" +
                                        "- turnright(a)" +
                                            " -> Turn the robot right with given angle \"a\" (degree).\n" +
                                        "- dash()" +
                                            " -> Dash the robot forward with x power.\n" +
                                        "- skill()" +
                                            " -> Activate the robot's special skill.\n" +
                                        "- clear()" +
                                            " -> Clear the command terminal.\n" +
                                        "- help" +
                                            " -> Show all available commands.\n";
        public static string EmptyCommandText = "> Empty command.\n";
        public static string UnknownCommandText = "> Unknown command!";
        public static string InvalidArgumentText = "> Invalid argument. Expected ";     // Need to implement the argument.
        public static string InvalidCommandText = "> Invalid command. Type \"help\" to show all commands";
        public static string TryExecuteCommandText = "> Try executing ";                // Need to implement the argument.
        public static string CommandExecutedText = "executed.";                         // Implement the command name. 
    }
    #endregion

}
