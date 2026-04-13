using System;
using System.Collections.Generic;
using System.Linq;
using SumoBot;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class BotUtility
{
    private static List<Type> cached;
    private static List<Bot> cachedBotInstances;

    public static List<Type> GetAllBotTypes()
    {
        if (cached != null && cached.Count > 0)
            return cached;

        try
        {
            var assemblies = AppDomain.CurrentDomain.GetAssemblies();
            Debug.Log($"[BotUtility] Scanning {assemblies.Length} assemblies for Bot types");

            cached = assemblies
                    .SelectMany(a =>
                    {
                        try
                        {
                            return a.GetTypes();
                        }
                        catch (System.Exception ex)
                        {
                            Debug.LogWarning($"[BotUtility] Failed to get types from assembly {a.FullName}: {ex.Message}");
                            return new Type[0];
                        }
                    })
                    .Where(t => typeof(Bot).IsAssignableFrom(t)
                                && !t.IsAbstract
                                && t.IsSubclassOf(typeof(ScriptableObject)))
                    .OrderBy(t => t.Name)
                    .ToList();

            Debug.Log($"[BotUtility] Found {cached.Count} Bot types");
            foreach (var type in cached)
            {
                Debug.Log($"[BotUtility]   - {type.FullName}");
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[BotUtility] Failed to get bot types: {ex.Message}");
            cached = new List<Type>();
        }

        return cached;
    }

    public static List<Bot> GetAllBotInstances()
    {
        if (cachedBotInstances != null && cachedBotInstances.Count > 0)
            return cachedBotInstances;

        cachedBotInstances = new List<Bot>();

        // In Build: Try loading from Resources folder
        // Search in "Bot" subfolder and root
        var resourceBots = Resources.LoadAll<Bot>("Bot").ToList();
        if (resourceBots.Count == 0)
        {
            resourceBots = Resources.LoadAll<Bot>("").ToList();
        }

        cachedBotInstances = resourceBots;

        // Fallback: create from types if no resources found
        if (cachedBotInstances.Count == 0)
        {
            Debug.LogWarning("[BotUtility] No Bot assets found in Resources folder. Creating instances from code...");

            var botTypes = GetAllBotTypes();
            Debug.Log($"[BotUtility] Found {botTypes.Count} bot types via reflection");

            foreach (var type in botTypes)
            {
                try
                {
                    Bot bot = ScriptableObject.CreateInstance(type) as Bot;
                    if (bot != null)
                    {
                        // Initialize the bot if needed
                        bot.name = type.Name;
                        cachedBotInstances.Add(bot);
                        Debug.Log($"[BotUtility] Created instance of {type.Name}");
                    }
                }
                catch (System.Exception ex)
                {
                    Debug.LogError($"[BotUtility] Failed to create instance of {type.Name}: {ex.Message}");
                }
            }

            if (cachedBotInstances.Count == 0)
            {
                Debug.LogError("[BotUtility] Failed to create any bot instances. Make sure Bot ScriptableObject assets exist in a Resources folder.");
            }
        }
        else
        {
            Debug.Log($"[BotUtility] Loaded {cachedBotInstances.Count} bot assets from Resources");
        }

        // Sort by ID
        cachedBotInstances = cachedBotInstances.OrderBy(bot => bot.ID).ToList();

        return cachedBotInstances;
    }
}