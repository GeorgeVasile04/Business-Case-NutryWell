# Data Analysis Summary: Parent Company Diagnostics (Course 2)

**Context:** This document outlines the diagnostic analysis performed on the **Parent Company's** operational data (the company that was recently acquired by NutrYWell). We applied the exact same mathematical logic used in Course 1 but tailored the entry lag calculation to the company's "J+1" target. We also added a geographic classification to understand their main markets.

---

## 0. New Metric: Order Classification by Country
**What we analyzed:** The distribution of order volumes across the European market.
**How it was done:** We counted the number of unique `Order IDs` delivered to each country and ranked them by their percentage of the total order volume.
**Top 10 Results:**
1. **ES (Spain):** 7.23% (224 orders)
2. **IT (Italy):** 6.00% (186 orders)
3. **DE (Germany):** 5.87% (182 orders)
4. **PT (Portugal):** 5.68% (176 orders)
5. **FR (France):** 5.36% (166 orders)
6. **PL (Poland):** 4.26% (132 orders)
7. **SE (Sweden):** 4.23% (131 orders)
8. **BE (Belgium):** 4.23% (131 orders)
9. **AT (Austria):** 4.19% (130 orders)
10. **DK (Denmark):** 3.65% (113 orders)

*(Note: The volume is heavily fragmented across 27 different countries, none of which capture more than 8% of the total network).*

---

## 1. Order Entry & Information Flow Analysis (J+1 Policy)
**What we analyzed:** The administrative processing time, but this time factoring in the official business rule: an order has a 1-day tolerance (J+1) to be entered into the ERP.
**Mathematical Computation:** `Entry Lag Days = ERP Entry Date - Order Date`. We then calculated the percentage of orders strictly `> 1 day`.
**Results:**
* Total unique orders evaluated: 3,099
* Average Lag: 1.65 days
* **Real Delay Percentage (> 1 day lag): 28.27%**

*Comparison to Course 1:* While the average lag is similar (about 1.6 days), applying the explicit business rule shows that roughly 28% of orders are actually failing the J+1 SLA, compared to the 56% that experienced "any" delay in the uncalibrated model.

---

## 2. Order Splitting & Digital Silos
**What we analyzed:** Whether the parent company naturally split customer baskets into multiple redundant shipments prior to the NutrYWell integration.
**Mathematical Computation:** Grouped by `Customer ID` and `Order Date`.
**Results:**
* Total unique Customer-Day Baskets: 3,075
* Total Baskets split into multiple Order IDs: 24
* **Split Percentage: 0.78%**

*Comparison to Course 1:* **This is a massive finding.** In Course 1, we found a 40.59% order split rate. Here, in the parent company's isolated network, it is less than 1%. This definitively proves that the extreme order splitting is an artificial consequence of the *acquisition and unintegrated IT systems*, NOT the natural buying behavior of the customers. 

---

## 3. Delivery Reliability & Service Level (OTIF)
**What we analyzed:** On-Time Delivery performance (`Actual Receipt Date <= Customer Requested Date`).
**Results:**
* Global On-Time vs Requested Date: 77.70%
* By Channel Breakdown:
  * Pharmacy: 79.81%
  * Retail: 81.11%
  * Retail Sport: 77.57%
  * **E-commerce: 75.28%**

*Comparison to Course 1:* The parent company's E-commerce OTIF was 75.28%. After the integration (Course 1), E-commerce OTIF collapsed to 49.20%. This suggests the parent company's network handled DTC (Direct-To-Consumer) much better than the combined transitional network does currently.

---

## 4. Logistics Inefficiencies and Network Fragmentation
**What we analyzed:** Logistics load optimization by calculating pallet utilization.
**Mathematical Computation:** `Ceiling(Total Weight / Min Pallet Capacity of 200kg)`.
**Results:**
* Total Physical Shipments (invoiced): 3,099  
* Total physical pallets paid for: 3,104
* Average Weight per Shipment: 31.91 kg (Max capacity: 200kg)
* **Overall Pallet Fill Rate: 15.95%**
* **Shipments weighing < 50 kg: 2368 (76.41%)**

*Comparison to Course 1:* Structural LTL inefficiency was already an issue. Over 76% of their historical shipments weighed less than 50kg, achieving only a ~16% fill rate. This means that while the recent IT-forced order splitting exacerbated the problem, the basic strategy of shipping tiny consumer orders via heavy pallet-centric retail logistics carriers is a historical flaw they brought into the merger.
