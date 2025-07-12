using System;
using System.Collections.Generic;
using System.Linq;
using SumoBot;
using UnityEngine;

public class BotUtility
{
    public static List<Type> GetAllBotTypes()
    {
        return AppDomain.CurrentDomain.GetAssemblies()
            .SelectMany(assembly => assembly.GetTypes())
            .Where(type => typeof(Bot).IsAssignableFrom(type) && !type.IsAbstract && type.IsSubclassOf(typeof(ScriptableObject)))
            .OrderBy(t => t.Name)
            .ToList();
    }
}