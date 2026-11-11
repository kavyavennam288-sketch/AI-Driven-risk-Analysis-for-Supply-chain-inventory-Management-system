
package com.supplychain.backend.service;

import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import java.util.Map;

@Service
public class AiRiskService {

    private final RestClient restClient;

    public AiRiskService() {
        this.restClient = RestClient.builder()
                .baseUrl("http://127.0.0.1:5000")
                .build();
    }

    public Map<?, ?> predictRisk(Map<String, Object> orderData) {

        return restClient.post()
                .uri("/api/predict-risk")
                .contentType(MediaType.APPLICATION_JSON)
                .body(orderData)
                .retrieve()
                .body(Map.class);
    }
}
