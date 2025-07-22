using System.Collections.Generic;
using SumoCore;
using SumoManager;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace SumoInput
{
    public class ButtonInputHandler : MonoBehaviour
    {
        #region UI Elements properties
        public ButtonPointerHandler Accelerate;
        public ButtonPointerHandler TurnLeft;
        public ButtonPointerHandler TurnRight;
        public ButtonPointerHandler Dash;
        public ButtonPointerHandler Skill;
        private InputProvider inputProvider;
        #endregion

        #region Runtime properties
        private readonly Dictionary<ActionType, float?> actionLastUsedMap = new();
        private readonly Dictionary<ActionType, InputType?> actionInputTypeMap = new();
        private Dictionary<ActionType, GameObject> actionButtonMap = new();

        #endregion

        #region Unity methods
        void Awake()
        {
            inputProvider = gameObject.GetComponent<InputProvider>();

            if (Accelerate == null || TurnLeft == null || TurnRight == null || Dash == null || Skill == null)
            {
                Debug.LogError("One or more ButtonPointerHandler references are not assigned in the Inspector!");
                return;
            }

            actionButtonMap = new Dictionary<ActionType, GameObject>
        {
            // Note: We use .gameObject because Accelerate, TurnLeft, etc. are ButtonPointerHandler components,
            // and we need the GameObject they are attached to.
            // { ActionType.Accelerate, Accelerate.gameObject },
            { ActionType.Accelerate, Accelerate.gameObject },
            { ActionType.TurnLeft, TurnLeft.gameObject },
            // { ActionType.TurnLeftWithAngle, TurnLeft.gameObject },
            { ActionType.TurnRight, TurnRight.gameObject },
            // { ActionType.TurnRightWithAngle, TurnRight.gameObject },
            { ActionType.Dash, Dash.gameObject },
            { ActionType.SkillBoost, Skill.gameObject },
            { ActionType.SkillStone, Skill.gameObject },
        };
        }

        void OnEnable()
        {
            Accelerate.Events[ButtonPointerHandler.OnHold].Subscribe(inputProvider.OnAccelerateButtonPressed);
            TurnLeft.Events[ButtonPointerHandler.OnHold].Subscribe(inputProvider.OnTurnLeftButtonPressed);
            TurnRight.Events[ButtonPointerHandler.OnHold].Subscribe(inputProvider.OnTurnRightButtonPressed);

            Dash.Events[ButtonPointerHandler.OnPress].Subscribe(inputProvider.OnDashButtonPressed);
            Skill.Events[ButtonPointerHandler.OnPress].Subscribe(inputProvider.OnSkillButtonPressed);

            BattleManager.Instance.Events[BattleManager.OnBattleChanged].Subscribe(OnBattleChanged);
            //SetUpButtonGuide();
        }

        void OnDisable()
        {
            Accelerate.Events[ButtonPointerHandler.OnHold].Unsubscribe(inputProvider.OnAccelerateButtonPressed);
            TurnLeft.Events[ButtonPointerHandler.OnHold].Unsubscribe(inputProvider.OnTurnLeftButtonPressed);
            TurnRight.Events[ButtonPointerHandler.OnHold].Unsubscribe(inputProvider.OnTurnRightButtonPressed);

            Dash.Events[ButtonPointerHandler.OnPress].Unsubscribe(inputProvider.OnDashButtonPressed);
            Skill.Events[ButtonPointerHandler.OnPress].Unsubscribe(inputProvider.OnSkillButtonPressed);

            BattleManager.Instance.Events[BattleManager.OnBattleChanged].Unsubscribe(OnBattleChanged);
        }

        void Update()
        {
            foreach (var item in actionLastUsedMap)
            {
                if (item.Value != null)
                {
                    bool isHolding = Time.time - item.Value < BattleManager.Instance.ActionInterval;
                    UpdateButtonState(item.Key, isHolding);
                }
            }

            if (BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing)
            {
                Battle battle = BattleManager.Instance.Battle;
                SumoController player = inputProvider.PlayerSide == PlayerSide.Left ? battle.LeftPlayer : battle.RightPlayer;

                Skill.GetComponentInChildren<Button>().interactable = !player.Skill.IsSkillOnCooldown;
                Dash.GetComponentInChildren<Button>().interactable = !player.IsDashOnCooldown;
            }
        }
        #endregion

        #region Button handling methods

        void OnBattleChanged(EventParameter param)
        {
            var battle = param.Battle;
            SumoController currentPlayer = inputProvider.PlayerSide == PlayerSide.Left ? BattleManager.Instance.Battle.LeftPlayer : BattleManager.Instance.Battle.RightPlayer;

            if (BattleManager.Instance.CurrentState == BattleState.Battle_Countdown)
                currentPlayer.Events[SumoController.OnAction].Subscribe(OnPlayerAction);
            else if (BattleManager.Instance.CurrentState == BattleState.Battle_End)
            {
                currentPlayer.Events[SumoController.OnAction].Unsubscribe(OnPlayerAction);
                Dash.GetComponentInChildren<Button>().interactable = true;
                Skill.GetComponentInChildren<Button>().interactable = true;
            }
        }

        void OnPlayerAction(EventParameter param)
        {
            ISumoAction action = param.Action;
            bool isPostExecute = param.Bool == false;

            if (isPostExecute)
            {
                actionLastUsedMap[action.Type] = Time.time;
                actionInputTypeMap[action.Type] = action.InputUsed;
            }
        }

        // private void SetUpButtonGuide()
        // {
        //     foreach (var item in InputProvider.KeyboardBindings[inputProvider.PlayerSide])
        //     {
        //         GameObject go = actionButtonMap[item.Value.Type];
        //         if (go != null)
        //         {
        //             TMP_Text text = go.GetComponentInChildren<TMP_Text>();
        //             text.SetText($"{go.name}\n({item.Key})");
        //         }
        //     }
        // }

        void UpdateButtonState(ActionType actionType, bool active)
        {
            GameObject buttonObject = actionButtonMap[actionType];
            InputType? inputType = actionInputTypeMap[actionType];

            Button button = buttonObject.GetComponent<Button>();

            if (inputType == null)
            {
                button.interactable = true;
                inputProvider.StateKeyboardAction[actionType] = true;
                return;
            }

            bool targetState = !active;

            if (inputType == InputType.Keyboard)
                button.interactable = targetState;
            else if (inputType == InputType.UI)
                inputProvider.StateKeyboardAction[actionType] = targetState;
            else
            {
                inputProvider.StateKeyboardAction[actionType] = targetState;
                button.interactable = targetState;
            }
        }
        #endregion
    }
}