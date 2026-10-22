using System;

namespace SumoServices
{
    /// <summary>
    /// A purchasable AI bot script (Type == "BotScript" in items.json). The only field
    /// unique to bot scripts today is WinRate — the common Creator/Description now live on
    /// CatalogItem so any category can carry them. Future bot-only fields (e.g. the runtime
    /// strategy this item maps to, see task-6) also belong here.
    /// </summary>
    [Serializable]
    public class BotScriptItem : CatalogItem
    {
        /// <summary>0..1 win rate shown in the detail panel. Bot-script only.</summary>
        public float WinRate;
    }
}
