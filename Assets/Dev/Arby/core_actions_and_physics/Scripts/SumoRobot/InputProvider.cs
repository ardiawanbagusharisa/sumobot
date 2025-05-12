using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

namespace CoreSumoRobot
{

    public enum BattleInputType
    {
        UI,
        LiveCommand,
        Script,
    }


    public class InputProvider : MonoBehaviour
    {
        public bool IncludeKeyboard;
        public PlayerSide PlayerSide;
        public ERobotSkillType SkillType;

        private Queue<ISumoAction> CommandQueue = new Queue<ISumoAction>();

        public InputProvider(PlayerSide side, bool includeKeyboard = false)
        {
            PlayerSide = side;
            IncludeKeyboard = includeKeyboard;
        }

        void OnEnable()
        {
            CommandQueue = new Queue<ISumoAction>();
        }

        public List<ISumoAction> GetInput()
        {
            var actions = new List<ISumoAction>();

            if (IncludeKeyboard)
            {
                actions = ReadKeyboardInput();
            }

            while (CommandQueue.Count > 0)
                actions.Add(CommandQueue.Dequeue());

            return actions;
        }

        #region Live Command / AI Script Input
        // APplied for Live Command And AI Script
        public void EnqueueCommand(ISumoAction action)
        {
            CommandQueue.Enqueue(action);
        }
        #endregion

        #region Keyboard Input
        private List<ISumoAction> ReadKeyboardInput()
        {
            var actions = new List<ISumoAction>();

            if (PlayerSide == PlayerSide.Left)
            {
                if (Input.GetKey(KeyCode.W))
                    actions.Add(new AccelerateAction());
                if (Input.GetKeyDown(KeyCode.LeftShift))
                    actions.Add(new DashAction());
                if (Input.GetKey(KeyCode.D))
                    actions.Add(new TurnRightAction());
                if (Input.GetKey(KeyCode.A))
                    actions.Add(new TurnLeftAction());
                if (Input.GetKeyDown(KeyCode.C))
                    actions.Add(new SkillAction(SkillType));
            }
            else
            {
                if (Input.GetKey(KeyCode.O))
                    actions.Add(new AccelerateAction());
                if (Input.GetKeyDown(KeyCode.RightShift))
                    actions.Add(new DashAction());
                if (Input.GetKey(KeyCode.Semicolon))
                    actions.Add(new TurnRightAction());
                if (Input.GetKey(KeyCode.K))
                    actions.Add(new TurnLeftAction());
                if (Input.GetKeyDown(KeyCode.M))
                    actions.Add(new SkillAction(SkillType));
            }
            return actions;
        }
        #endregion

        #region UI Input
        public void OnAccelerateButtonPressed()
        {
            CommandQueue.Enqueue(new AccelerateAction());
        }
        public void OnDashButtonPressed()
        {
            CommandQueue.Enqueue(new DashAction());
        }

        public void OnTurnLeftButtonPressed()
        {
            CommandQueue.Enqueue(new TurnLeftAction());
        }

        public void OnTurnRightButtonPressed()
        {
            CommandQueue.Enqueue(new TurnRightAction());
        }

        public void OnBoostSkillButtonPressed()
        {
            if (SkillType != ERobotSkillType.Boost) return;
            CommandQueue.Enqueue(new SkillAction(ERobotSkillType.Boost));
        }

        public void OnStoneSkillButtonPressed()
        {
            if (SkillType != ERobotSkillType.Stone) return;
            CommandQueue.Enqueue(new SkillAction(ERobotSkillType.Stone));
        }
        #endregion
    }

}
