using System;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace SumoServices
{
    /// <summary>
    /// Picks the concrete CatalogItem subtype to deserialize each items.json row into,
    /// based on the "Type" field (e.g. "BotScript" -> BotScriptItem). Unknown/empty Type
    /// falls back to the base CatalogItem so existing rows (Skin, Visual Script, Module)
    /// don't need a Type value.
    /// </summary>
    public class CatalogItemConverter : JsonConverter<CatalogItem>
    {
        public override CatalogItem ReadJson(JsonReader reader, Type objectType, CatalogItem existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            JObject obj = JObject.Load(reader);
            string type = obj["Type"]?.Value<string>();

            CatalogItem item;
            if (string.Equals(type, "BotScript", StringComparison.OrdinalIgnoreCase))
                item = new BotScriptItem();
            else if (string.Equals(type, "Skin", StringComparison.OrdinalIgnoreCase))
                item = new SkinItem();
            else
                item = new CatalogItem();

            serializer.Populate(obj.CreateReader(), item);
            return item;
        }

        public override void WriteJson(JsonWriter writer, CatalogItem value, JsonSerializer serializer)
        {
            serializer.Serialize(writer, value, value.GetType());
        }
    }
}
