using UnityEngine;
using System.Collections.Generic;
using SumoInput;
using SumoCore;
using SumoManager;
using System;
using System.Linq;

namespace SumoBot.EA.MCTS
{

    [Serializable]
    public class AI_MCTS_Config
    {
        public int ResetIterationAt = 2;
        public float Iterations = 100;
        public float UCBConstant = 1.41f;
        public float ScoreLimit = -300;
        public string Name = "MCTS Example";
        public SkillType DefaultSkillType = SkillType.Boost;
    }

    public class AIBot_EA_MCTS : Bot
    {
        public AI_MCTS_Config config = new();
        public override string ID => config.Name;
        public override SkillType SkillType => config.DefaultSkillType;


        #region Runtime properties
        private EA_MCTS_Node root;
        private List<ISumoAction> lastActionsFromEnemy;
        private List<ISumoAction> lastActionsToEnemy;
        public Dictionary<string, EA_MCTS_Node> AllNodes = new();
        public static List<ISumoAction> PossibleActions = new() {
            new TurnAction(InputType.Script, ActionType.TurnLeft, 0.1f),
            new TurnAction(InputType.Script, ActionType.TurnRight, 0.1f),

            new TurnAction(InputType.Script, ActionType.TurnLeft, 0.3f),
            new TurnAction(InputType.Script, ActionType.TurnRight, 0.3f),

            new AccelerateAction(InputType.Script, 0.1f),
            new AccelerateAction(InputType.Script, 0.3f),
            new DashAction(InputType.Script),
            new SkillAction(InputType.Script),
        };
        private BattleState currState;
        private PlayerSide side;
        private int elapsedInterval = 0;
        #endregion

        private SumoAPI api;
        public override void OnBotUpdate()
        {
            elapsedInterval += 1;

            if (elapsedInterval >= config.ResetIterationAt)
            {
                elapsedInterval = 0;
                InitNode();
            }
            Decide();

            Submit();
        }

        public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
        {
            currState = state;

            if (currState == BattleState.Battle_Countdown)
            {
                lastActionsFromEnemy = null;
                lastActionsToEnemy = null;
                InitNode();
            }
        }

        public override void OnBotCollision(BounceEvent bounceEvent)
        {
            if (side == bounceEvent.Actor)
                lastActionsFromEnemy = null;
            else
                lastActionsToEnemy = null;
            ClearCommands();
            InitNode();
        }

        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;
            InitNode();
        }

        #region Custom Functions
        private void InitNode()
        {
            AllNodes.Clear();
            root = new EA_MCTS_Node(
                null,
                new List<ISumoAction>(PossibleActions),
                goodAction: lastActionsToEnemy,
                badAction: lastActionsFromEnemy)
            {
                Side = side,
                ID = "Root"
            };
            root.Init(AllNodes);
        }

        EA_MCTS_Node Decide()
        {
            Debug.Log($"testing {api.Angle(normalized: true)}");
            for (int i = 0; i < config.Iterations; i++)
            {
                EA_MCTS_Node selected = root.Select(config);
                var expanded = selected.Expand(AllNodes);
                if (expanded != null)
                {
                    var result = expanded.Simulate(api, config);
                    expanded.Backpropagate(result);
                }

            }
            EA_MCTS_Node bestChild = root.GetBestChild();
            if (bestChild == null)
                return null;

            if (bestChild.totalReward <= config.ScoreLimit)
            {
                Debug.Log($"[AIBot_EA_MCTS] LowestScoreToReInit reached {bestChild.totalReward}");
                InitNode();
                return null;
            }

            // Debug.Log($"[AIBot_EA_MCTS] selected-score: {bestChild.totalReward}, selected-action(s): {bestChild.ID} selected-visits: {bestChild.visits}, {string.Join("->", bestChild.actions.Select((x) => x.FullName))}");

            lastActionsToEnemy = bestChild.actions;

            foreach (var act in bestChild.actions)
            {
                // act.Reason = api.GenerateReason(act);
                // if (act is TurnAction && api.IsActionActive(act))
                //     continue;
                Enqueue(act);
            }
            return bestChild;
        }

        #endregion
    }

    public class EA_MCTS_Node
    {
        public string ID;
        public EA_MCTS_Node head;
        public EA_MCTS_Node parent;
        public List<EA_MCTS_Node> children = new();
        public int visits = 0;
        public float totalReward = 0f;
        public float angleScore = 0f;
        public float distScore = 0f;
        public float bonusOrPenalty = 0f;
        public List<ISumoAction> actions = new();
        public string actionString;
        public List<ISumoAction> badAction;
        public List<ISumoAction> goodAction;
        public bool badActionAlreadyUsed = false;
        public bool goodActionAlreadyUsed = false;
        public PlayerSide Side;

        public EA_MCTS_Node(EA_MCTS_Node parent, List<ISumoAction> actions, List<ISumoAction> goodAction = null, List<ISumoAction> badAction = null)
        {
            this.parent = parent;
            this.actions = actions;
            this.badAction = badAction;
            this.goodAction = goodAction;
        }

        public void Init(Dictionary<string, EA_MCTS_Node> AllNodes)
        {
            children.Clear();
            totalReward = 0f;
            visits = 0;
            foreach (var action in actions)
            {
                EA_MCTS_Node newNode = new(this, new List<ISumoAction>() { action });
                newNode.ID = action.FullName;
                if (goodAction != null && goodAction.Count > 0)
                {
                    newNode.totalReward = 1;
                    newNode.visits = 1;
                }
                if (badAction != null && badAction.Count > 0)
                {
                    newNode.totalReward = -1;
                    newNode.visits = 1;
                }
                children.Add(newNode);
                AllNodes.Add(action.FullName, newNode);
            }
        }

        public EA_MCTS_Node Expand(Dictionary<string, EA_MCTS_Node> AllNodes)
        {
            var unexploredActs = AIBot_EA_MCTS.PossibleActions.Where(x =>
            {
                var newActNames = $"{ID}:{x.FullName}";
                if (!AllNodes.ContainsKey(newActNames))
                {
                    return true;
                }
                return false;
            }).ToList();

            if (unexploredActs.Count() == 0)
                return null;

            System.Random random = new();
            var randomAction = unexploredActs[random.Next(unexploredActs.Count())];
            string newActNames;
            newActNames = $"{ID}:{randomAction.FullName}";

            var newActs = new List<ISumoAction>(actions) { randomAction };

            EA_MCTS_Node newNode = new(this, newActs)
            {
                ID = newActNames
            };
            children.Add(newNode);
            AllNodes.Add(newActNames, newNode);
            return newNode;
        }

        public void SortAction()
        {
            actions.Sort((x, y) =>
            {
                if (x.FullName.ToLower().Contains("turn"))
                {
                    return -100;
                }
                return x.FullName.GetHashCode().CompareTo(y.FullName.GetHashCode());
            });
        }

        public EA_MCTS_Node Select(AI_MCTS_Config config)
        {
            if (children.Count == 0) return this;
            float C = config.UCBConstant;

            return children.OrderByDescending(child =>
            {
                if (child.visits == 0) return double.MaxValue;
                float exploitation = child.totalReward / child.visits;
                double exploration = C * Math.Sqrt(Math.Log(visits + 1) / child.visits);
                return exploitation + exploration;
            }).FirstOrDefault();
        }

        public (float, float, float) Simulate(SumoAPI api, AI_MCTS_Config config)
        {

            float bonusOrPenalty = 0;
            float angleScore = 0;
            float distScore = 0;

            foreach (var action in actions)
            {
                List<string> reason = new();
                Vector2 predPos;
                float predRot;

                (predPos, predRot) = api.Simulate(new() { action });

                bool approachWithPosition = action is AccelerateAction || action is DashAction;
                float preAngleScore = api.Angle(oriPos: predPos, oriRot: predRot, normalized: true);
                float preDistScore = api.DistanceNormalized(oriPos: predPos);

                angleScore += preAngleScore;
                distScore += preDistScore;

                if (action.Type == ActionType.SkillStone)
                {
                    if (api.CanExecute(action))
                    {
                        bonusOrPenalty += 10f;
                        reason.Add("Use skill stone immediately");
                    }
                    else
                        bonusOrPenalty -= 0.1f;
                    // continue;
                }
                else if (action.Type == ActionType.SkillBoost)
                {
                    if (api.CanExecute(action))
                        bonusOrPenalty += 10f;
                    else
                        bonusOrPenalty -= 0.1f;
                    // continue;
                }
                else if (action.Type == ActionType.Dash)
                {
                    if (!api.CanExecute(action))
                        bonusOrPenalty -= 0.1f;
                    // continue;
                }

                if (action is TurnAction)
                {
                    if (api.Angle(normalized: true) > preAngleScore)
                    {
                        bonusOrPenalty -= 5f;
                    }
                }
                if (action is AccelerateAction)
                {

                    if (preAngleScore > 0.85f)
                    {
                        bonusOrPenalty += 3f;
                    }
                }
                else if (action is DashAction && !api.MyRobot.IsDashOnCooldown)
                {
                    if (preAngleScore > 0.95f)
                    {
                        bonusOrPenalty += 5f;
                    }
                }

                Vector2 distanceFromArena = api.Distance(targetPos: api.BattleInfo.ArenaPosition, oriPos: predPos);

                if (distanceFromArena.magnitude >= api.BattleInfo.ArenaRadius)
                {
                    bonusOrPenalty -= 100f;
                }
            }

            float normAngleScore = angleScore / actions.Count();
            float normDistScore = distScore / actions.Count();
            float normBonusOrPenalty = bonusOrPenalty / actions.Count();

            this.angleScore += normAngleScore;
            this.distScore += normBonusOrPenalty;
            this.bonusOrPenalty += normBonusOrPenalty;
            return (normAngleScore, normDistScore, normBonusOrPenalty);
        }

        public void Backpropagate((float, float, float) reward)
        {
            visits++;
            totalReward += reward.Item1 + reward.Item2 + reward.Item3;
            angleScore += reward.Item1;
            distScore += reward.Item2;
            bonusOrPenalty += reward.Item3;
            parent?.Backpropagate(reward);
        }

        public EA_MCTS_Node GetBestChild(List<EA_MCTS_Node> nodes = null)
        {
            if (children.Count == 0) return null;

            var highest = children.OrderByDescending(child =>
            {
                double exploitation = child.totalReward / (child.visits + double.Epsilon);
                return exploitation;
            }).FirstOrDefault();

            // return highest;

            if (nodes == null)
            {
                nodes = new() { highest };
            }
            else
            {
                nodes.Add(highest);
            }

            if (highest.children.Count == 0)
            {
                if (nodes == null)
                    return highest;
                else
                {
                    return nodes.OrderByDescending(child =>
                    {
                        double exploitation = child.totalReward / (child.visits + double.Epsilon);
                        return exploitation;
                    }).FirstOrDefault();
                }
            }
            return highest.GetBestChild(nodes);
        }
    }
}