using System;
using SumoBot;
using SumoCore;
using SumoInput;
using Unity.VisualScripting;
using UnityEngine;

namespace SumoManager
{
    public class InputManager : MonoBehaviour
    {
        public static InputManager Instance { get; private set; }

        #region Input properties
        public GameObject LeftButton;
        public GameObject RightButton;
        public GameObject LeftLiveCommand;
        public GameObject RightLiveCommand;
        #endregion

        #region Runtimep properties
        private BotManager botManager;
        #endregion

        #region Unity methods
        private void Awake()
        {
            if (Instance != null)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
        }

        void OnEnable()
        {
            botManager = GetComponent<BotManager>();
        }
        #endregion

        #region Input methods
        public void InitializeInput(SumoController controller)
        {

            GameObject liveCommandObject = controller.Side == PlayerSide.Left ? LeftLiveCommand : RightLiveCommand;
            GameObject UIButtonsObject = controller.Side == PlayerSide.Left ? LeftButton : RightButton;

            GameObject selectedInputObject;

            switch (BattleManager.Instance.BattleInputType)
            {
                case InputType.Script:
                    liveCommandObject.SetActive(false);
                    UIButtonsObject.SetActive(false);
                    selectedInputObject = null;
                    break;

                case InputType.LiveCommand:
                    liveCommandObject.SetActive(true);
                    selectedInputObject = liveCommandObject;
                    UIButtonsObject.SetActive(false);
                    break;

                // UI button & keyboard 
                default:
                    UIButtonsObject.SetActive(true);
                    selectedInputObject = UIButtonsObject;
                    liveCommandObject.SetActive(false);
                    break;
            }

            SumoAPI api = CreateAPI(controller.Side);
            InputProvider inputProvider = GetInputProvider(controller, selectedInputObject);

            controller.InputProvider = inputProvider;

            // Additional initialization
            switch (BattleManager.Instance.BattleInputType)
            {
                case InputType.UI:
                case InputType.Keyboard:
                    // Enable for test-only
                    SetupBots(controller, api);
                    break;
                case InputType.Script:
                    SetupBots(controller, api);
                    break;
                case InputType.LiveCommand:
                    // Enable for test-only
                    SetupBots(controller, api);
                    liveCommandObject.GetComponent<CommandSystem>().InitCommandSystem(api);
                    break;
            }
        }

        private InputProvider GetInputProvider(SumoController controller, GameObject selectedInputObject)
        {
            InputType battleInputType = BattleManager.Instance.BattleInputType;
            InputProvider inputProvider;
            if (battleInputType == InputType.Script)
            {
                if (GetComponent<BotManager>().IsEnable)
                {
                    InputProvider scriptInputProvider = controller.AddComponent<InputProvider>();
                    scriptInputProvider.PlayerSide = controller.Side;
                    scriptInputProvider.IncludeKeyboard = false;
                    inputProvider = scriptInputProvider;
                }
                else
                    throw new Exception($"Battle with [{battleInputType}] should provide InputProvider");
            }
            else
            {
                if (selectedInputObject == null)
                    throw new Exception($"One of [{battleInputType}]'s object must be used");

                inputProvider = selectedInputObject.GetComponent<InputProvider>();
            }

            return inputProvider;
        }

        private SumoAPI CreateAPI(PlayerSide side)
        {
            SumoController leftPlayer = BattleManager.Instance.Battle.LeftPlayer;
            SumoController rightPlayer = BattleManager.Instance.Battle.RightPlayer;

            SumoAPI api;

            if (side == PlayerSide.Left)
            {
                api = new SumoAPI(leftPlayer, rightPlayer);
            }
            else
            {
                api = new SumoAPI(rightPlayer, leftPlayer);
            }
            return api;
        }

        private void SetupBots(SumoController controller, SumoAPI api)
        {
            if (!botManager.IsEnable) return;

            SumoController leftPlayer = BattleManager.Instance.Battle.LeftPlayer;
            SumoController rightPlayer = BattleManager.Instance.Battle.RightPlayer;

            // Handle if the Bot is attached to left and right player (MonoBehaviour)
            if (!botManager.IsScriptable)
            {
                botManager.Left = leftPlayer.GetComponentInChildren<Bot>();
                botManager.Right = rightPlayer.GetComponentInChildren<Bot>();
            }

            if (botManager.Left != null && controller.Side == PlayerSide.Left)
            {
                botManager.Left.SetProvider(controller.InputProvider);
                botManager.Left.OnBotInit(controller.Side, api);
                leftPlayer.Actions[SumoController.OnPlayerBounce].Subscribe(botManager.Left.OnBotCollision);
            }
            else if (botManager.Right != null && controller.Side == PlayerSide.Right)
            {
                botManager.Right.SetProvider(controller.InputProvider);
                botManager.Right.OnBotInit(controller.Side, api);
                rightPlayer.Actions[SumoController.OnPlayerBounce].Subscribe(botManager.Right.OnBotCollision);
            }
        }
        #endregion
    }
}