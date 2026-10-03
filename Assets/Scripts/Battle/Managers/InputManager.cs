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

        #region Runtime properties
        private BotManager botManager;
        private PacingManager pacingManager;
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
            pacingManager = GetComponent<PacingManager>();
        }
        #endregion

        #region Input methods
        public void InitializeInput(
            SumoController controller,
            InputType type,
            bool initializeSimulationSystems = true)
        {
            GameObject liveCommandObject = controller.Side == PlayerSide.Left ? LeftLiveCommand : RightLiveCommand;
            GameObject UIButtonsObject = controller.Side == PlayerSide.Left ? LeftButton : RightButton;

            GameObject selectedInputObject;

            switch (type)
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

            InputProvider inputProvider = GetInputProvider(controller, selectedInputObject, type);
            inputProvider.API = api;
            controller.InputProvider = inputProvider;

            // The authoritative host needs bot/pacing systems. Online clients
            // only need an input provider for their local controls; initializing
            // pacing there also assumes host-only battle-log rounds exist.
            if (!initializeSimulationSystems)
            {
                // A remote-view client still needs its own Live Command terminal
                // initialized so parsed commands can be routed to the host. The
                // opponent's terminal is hidden and must not accept local input.
                if (type == InputType.LiveCommand &&
                    (!SumoMultiplayer.OnlineBattleSession.IsActive ||
                     controller.Side == SumoMultiplayer.OnlineBattleSession.LocalSide))
                {
                    liveCommandObject.GetComponent<CommandSystem>()?.InitCommandSystem(api);
                }
                return;
            }

            InitializePacing(controller);

            // Additional initialization
            switch (type)
            {
                case InputType.UI:
                case InputType.Keyboard:
                    // Test Onyl
                    botManager.Init(controller);
                    break;
                case InputType.Script:
                    botManager.Init(controller);
                    break;
                case InputType.LiveCommand:
                    // Test Onyl
                    botManager.Init(controller);
                    liveCommandObject.GetComponent<CommandSystem>().InitCommandSystem(api);
                    break;
            }
        }

        private InputProvider GetInputProvider(SumoController controller, GameObject selectedInputObject, InputType type)
        {
            InputProvider inputProvider;
            if (type == InputType.Script)
            {
                if (botManager.BotEnabled)
                {
                    InputProvider scriptInputProvider = controller.AddComponent<InputProvider>();
                    scriptInputProvider.PlayerSide = controller.Side;
                    scriptInputProvider.IncludeKeyboard = false;
                    inputProvider = scriptInputProvider;
                }
                else
                    throw new Exception($"Battle with [{type}] should provide InputProvider");
            }
            else
            {
                if (selectedInputObject == null)
                    throw new Exception($"One of [{type}]'s object must be used");

                inputProvider = selectedInputObject.GetComponent<InputProvider>();
            }
            inputProvider.InputType = type;

            return inputProvider;
        }

        private SumoAPI CreateAPI(PlayerSide side)
        {
            SumoController leftPlayer = BattleManager.Instance.Battle.LeftPlayer;
            SumoController rightPlayer = BattleManager.Instance.Battle.RightPlayer;

            SumoAPI api;

            if (side == PlayerSide.Left)
                api = new(leftPlayer, rightPlayer);
            else
                api = new(rightPlayer, leftPlayer);

            return api;
        }

        private void InitializePacing(SumoController controller)
        {
            if (pacingManager == null)
            {
                Logger.Warning($"[{controller.Side}] PacingManager not found, skipping pacing initialization");
                return;
            }

            if (!pacingManager.isActiveAndEnabled)
                return;

            pacingManager.Initialize(controller.Side, controller);
        }
        #endregion
    }
}
