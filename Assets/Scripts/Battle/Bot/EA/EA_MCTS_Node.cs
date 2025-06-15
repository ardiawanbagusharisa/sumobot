using System;
using System.Collections.Generic;
using System.Linq;
using CoreSumo;
using Unity.Collections;
using UnityEditor.Experimental.GraphView;
using UnityEngine;

namespace BotAI
{
    public class EA_MCTS_Node
    {
        public string ID;
        public EA_MCTS_Node head;
        public EA_MCTS_Node parent;
        public List<EA_MCTS_Node> children = new();
        public int visits = 0;
        public float totalReward = 0f;

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
                newNode.ID = action.Name;
                if (goodAction != null && goodAction.Count > 0)
                {
                    newNode.totalReward = 10;
                    newNode.visits = 1;
                }
                if (badAction != null && badAction.Count > 0)
                {
                    newNode.totalReward = -10;
                    newNode.visits = 1;
                }
                children.Add(newNode);
                AllNodes.Add(action.Name, newNode);
            }
        }

        public EA_MCTS_Node Expand(Dictionary<string, EA_MCTS_Node> AllNodes)
        {
            var unexploredActs = AIBot_EA_MCTS.PossibleActions.Where(x =>
            {
                var newActNames = $"{ID}:{x.Name}";
                if (!AllNodes.ContainsKey(newActNames))
                {
                    return true;
                }
                return false;
            }).ToList();

            if (unexploredActs.Count() == 0)
            {
                return null;
            }

            System.Random random = new();
            var randomAction = unexploredActs[random.Next(unexploredActs.Count())];
            string newActNames;
            newActNames = $"{ID}:{randomAction.Name}";

            var newActs = new List<ISumoAction>(actions)
                {
                    randomAction
                };

            EA_MCTS_Node newNode = new(this, newActs);
            newNode.ID = newActNames;
            children.Add(newNode);
            AllNodes.Add(newActNames, newNode);
            return newNode;
        }

        public EA_MCTS_Node Select()
        {
            if (children.Count == 0) return this;
            double C = 1.41;
            return children.OrderByDescending(child =>
            {
                if (child.visits == 0) return double.MaxValue;
                double exploitation = child.totalReward / child.visits;
                double exploration = C * Math.Sqrt(Math.Log(visits + 1) / child.visits);
                return exploitation + exploration;
            }).First();
        }

        public float Simulate(SumoController enemy, SumoController controller, float simulationTime)
        {
            GameObject arena = BattleManager.Instance.Arena;
            float arenaRadius = arena.GetComponent<CircleCollider2D>().radius * arena.transform.lossyScale.x;
            Vector3 arenaCenter = arena.transform.position;
            Vector3 aiDirection = controller.transform.up;
            Vector3 aiPosition = controller.transform.position;

            float bonusOrPenalty = 0;
            float angleScore = 0;
            float distScore = 0;

            foreach (var action in actions)
            {
                if (action is TurnLeftAngleAction)
                {
                    aiDirection += Quaternion.Euler(0, 0, (float)action.Param) * aiDirection * simulationTime * controller.TurnRate;
                }
                else if (action is TurnRightAngleAction rV)
                {
                    aiDirection += Quaternion.Euler(0, 0, (float)action.Param) * aiDirection * simulationTime * controller.TurnRate;
                }
                else if (action is AccelerateAction)
                {
                    aiPosition += aiDirection.normalized * controller.MoveSpeed * simulationTime;
                }
                else if (action is DashAction)
                {
                    if (controller.IsDashCooldown)
                    {
                        bonusOrPenalty += -0.1f;
                    }
                    else
                    {
                        bonusOrPenalty += 0.1f;
                        aiPosition += aiDirection.normalized * (controller.DashSpeed * controller.DashDuration) * controller.StopDelay * simulationTime;
                    }

                }
                else if (action is SkillAction)
                {
                    if (controller.Skill.IsSkillCooldown)
                    {
                        bonusOrPenalty += -0.5f;
                    }
                    else
                    {
                        if (controller.Skill.Type == ERobotSkillType.Boost)
                        {
                            bonusOrPenalty += 0.5f;
                            aiPosition += aiDirection.normalized * (controller.MoveSpeed * controller.Skill.BoostMultiplier) * simulationTime;
                        }
                    }
                }

                Vector3 toEnemy = enemy.transform.position - aiPosition;
                float distance = toEnemy.magnitude;
                float angle = Vector3.SignedAngle(aiDirection, toEnemy.normalized, Vector3.forward);

                angleScore += Mathf.Cos(angle * Mathf.Deg2Rad);
                distScore += 1f - Mathf.Clamp01(distance / arenaRadius);

                var distanceFromCenter = Vector3.Distance(aiPosition, arenaCenter);

                List<string> actionsInString = actions.Select((a) => a.Name.ToLower()).ToList();

                bool isActionIncludeAccelerating = actionsInString.Contains("accelerateaction") || actionsInString.Contains("dashaction") || actionsInString.Contains("skillaction");

                if ((distanceFromCenter > arenaRadius) && isActionIncludeAccelerating)
                {
                    // Penalize heavily if sumo will exits the ring, or any action that makes the Sumo move away from exits, reward instead.
                    bonusOrPenalty += arenaRadius - distanceFromCenter + (angleScore - 0.9f + distScore - 0.8f);
                    Debug.Log($"[Simulate][IsPossibleOutFromArena] {ID}, can cause go outside of arena\n Detail: {aiPosition}, {arenaCenter} > {arenaRadius}, resulting: {bonusOrPenalty}");
                }
            }

            return (angleScore / actions.Count()) + (distScore / actions.Count()) + (bonusOrPenalty / actions.Count());
        }

        public void Backpropagate(float reward)
        {
            visits++;
            totalReward += reward;
            parent?.Backpropagate(reward);
        }

        public EA_MCTS_Node GetBestChild()
        {
            if (children.Count == 0) return null;

            var highest = children.OrderByDescending(child =>
            {
                double exploitation = child.totalReward / (child.visits + double.Epsilon);
                return exploitation;
            }).First();
            return highest;

            if (highest.children.Count > 0)
            {
                // if (highestParam == null)
                // {
                //     highestParam = new() { highest };
                // }
                // else
                // {
                //     highestParam.Add(highest);
                // }
                return highest.GetBestChild();
            }

            return highest;
            //     if (highestParam == null)
            //     {
            //         return highest;
            //     }
            //     return highestParam.OrderByDescending(child =>
            //    {
            //        double exploitation = child.totalReward / child.visits;
            //        return exploitation;
            //    }).First();
        }
    }
}