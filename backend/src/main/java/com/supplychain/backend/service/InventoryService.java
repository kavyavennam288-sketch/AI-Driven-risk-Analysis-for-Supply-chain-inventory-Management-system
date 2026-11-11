
package com.supplychain.backend.service;

import com.supplychain.backend.entity.Inventory;
import com.supplychain.backend.repository.InventoryRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class InventoryService {

    private final InventoryRepository inventoryRepository;

    public InventoryService(InventoryRepository inventoryRepository) {
        this.inventoryRepository = inventoryRepository;
    }

    public List<Inventory> getAllInventory() {
        return inventoryRepository.findAll();
    }

    public double calculateSafetyStock(Inventory inventory) {

        double leadTimeDemand =
                inventory.getDailyDemand() * inventory.getLeadTimeDays();

        double riskScore = inventory.getSupplier() != null
                ? inventory.getSupplier().getRiskScore()
                : 0;

        double riskFactor;

        if (riskScore < 20) {
            riskFactor = 0.10;
        } else if (riskScore < 40) {
            riskFactor = 0.20;
        } else {
            riskFactor = 0.40;
        }

        return leadTimeDemand * riskFactor;
    }

    public double calculateReorderPoint(Inventory inventory) {

        double leadTimeDemand =
                inventory.getDailyDemand() * inventory.getLeadTimeDays();

        return leadTimeDemand + calculateSafetyStock(inventory);
    }

    public boolean needsReordering(Inventory inventory) {
        return inventory.getCurrentStock()
                <= calculateReorderPoint(inventory);
    }
}
