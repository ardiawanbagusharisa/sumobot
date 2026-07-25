using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SumoServices.Tests
{
    /// <summary>
    /// In-memory ICatalogService for unit tests: seeded with items in the constructor, no
    /// Resources / JSON. LoadAsync is a no-op success. Also implements IPlayerCatalogWriter so
    /// the publish flow can register items into the same read model the Market resolves against.
    /// </summary>
    public class FakeCatalogService : ICatalogService, IPlayerCatalogWriter
    {
        private readonly Dictionary<string, CatalogItem> byId = new();

        public FakeCatalogService(params CatalogItem[] items)
        {
            foreach (var item in items) byId[item.Id] = item;
        }

        public IReadOnlyList<CatalogItem> AllItems => byId.Values.ToList();

        public Task<ServiceResult> LoadAsync() => Task.FromResult(ServiceResult.Ok());

        public CatalogItem GetById(string itemId)
            => itemId != null && byId.TryGetValue(itemId, out var item) ? item : null;

        public ServiceResult Add(CatalogItem item)
        {
            if (item == null) return ServiceResult.Fail("item is required.");
            if (string.IsNullOrEmpty(item.Id)) return ServiceResult.Fail("item.Id is required.");
            if (byId.ContainsKey(item.Id)) return ServiceResult.Fail($"An item with id '{item.Id}' already exists.");
            byId[item.Id] = item;
            return ServiceResult.Ok();
        }

        public void Remove(string itemId)
        {
            if (!string.IsNullOrEmpty(itemId)) byId.Remove(itemId);
        }
    }
}
