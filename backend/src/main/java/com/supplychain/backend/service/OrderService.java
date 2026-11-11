
package com.supplychain.backend.service;

import com.supplychain.backend.entity.Order;
import com.supplychain.backend.repository.OrderRepository;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
public class OrderService {

    private final OrderRepository orderRepository;
    private final AiRiskService aiRiskService;

    public OrderService(OrderRepository orderRepository,
                        AiRiskService aiRiskService) {
        this.orderRepository = orderRepository;
        this.aiRiskService = aiRiskService;
    }

    public List<Order> getAllOrders() {
        return orderRepository.findAll();
    }

    public Order getOrderById(Long id) {
        return orderRepository.findById(id)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.NOT_FOUND, "Order not found with ID: " + id
                ));
    }

    public Map<?, ?> predictRiskForOrder(Long id) {
        Order order = getOrderById(id);

        Map<String, Object> orderData = new LinkedHashMap<>();

        orderData.put("Days for shipment (scheduled)", order.getShippingDaysScheduled());
        orderData.put("Order Item Discount", order.getDiscount());
        orderData.put("Order Item Discount Rate", order.getDiscountRate());
        orderData.put("Order Item Product Price", order.getProductPrice());
        orderData.put("Order Item Quantity", order.getQuantity());
        orderData.put("Sales", order.getSales());
        orderData.put("Order Item Total", order.getOrderTotal());
        orderData.put("Product Price", order.getProductPrice());
        orderData.put("Category Id", order.getCategoryId());
        orderData.put("Department Id", order.getDepartmentId());
        orderData.put("Latitude", order.getLatitude());
        orderData.put("Longitude", order.getLongitude());
        orderData.put("Type", order.getPaymentType());
        orderData.put("Category Name", order.getCategoryName());
        orderData.put("Customer Segment", order.getCustomerSegment());
        orderData.put("Market", order.getMarket());
        orderData.put("Order Region", order.getOrderRegion());
        orderData.put("Shipping Mode", order.getShippingMode());
        orderData.put("Department Name", order.getDepartmentName());

        Map<?, ?> prediction = aiRiskService.predictRisk(orderData);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("orderId", order.getId());
        result.put("orderNumber", order.getOrderNumber());
        prediction.forEach((key, value) ->
                result.put(String.valueOf(key), value)
        );

        return result;
    }
}
