using System;
using System.Collections.Generic;
using UnityEngine;

namespace RobotCoreAction
{
    public enum RobotInputType
    {
        Keyboard,
        UI,
        Script
    }

    public class CoreActionManager : MonoBehaviour
    {
        public static CoreActionManager Instance { get; private set; }
        public RobotInputType InputType = RobotInputType.Keyboard;
        public List<GameObject> playersList;
        private Dictionary<string, GameObject> players = new Dictionary<string, GameObject>();


        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
            }
            else
            {
                Destroy(gameObject);
            }
        }

        private void Start()
        {
            Init();
        }

        private void Init()
        {
            PreparePlayers();
        }

        private void PreparePlayers()
        {
            foreach (var player in playersList)
            {
                var playerId = player.GetComponent<CoreActionRobot>().Id;
                if (!players.ContainsKey(playerId))
                {
                    switch (InputType)
                    {
                        case RobotInputType.Keyboard:
                            player.GetComponent<CoreActionRobotController>().UseInput(new KeyboardInputProvider());

                            break;
                        case RobotInputType.UI:
                            player.GetComponent<CoreActionRobotController>().UseInput(new UIInputProvider());

                            break;
                        case RobotInputType.Script:
                            player.GetComponent<CoreActionRobotController>().UseInput(new ScriptInputProvider());
                            break;
                    }
                    players.Add(playerId, player);
                }
            }
        }
    }
}