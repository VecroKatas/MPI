#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <chrono>

#include "dijkstra.h"
#include "map.h"

static const auto INF = std::numeric_limits<int>::max();

void dijkstra(const Map& m, const std::string& initialNodeName, const std::string& goalNodeName) {
    auto start = std::chrono::high_resolution_clock::now();

    const auto& weights = m.getWeights();
    const auto& nodesNames = m.getNodesNames();
    auto nodesCount = nodesNames.size();

    std::vector<int> nodes(nodesCount);
    std::vector<int> distances(nodesCount);
    std::vector<int> prevNodes(nodesCount);

    std::set<int> unvisited;
    for (auto node = 0u; node < nodesCount; ++node) {
        unvisited.insert(node);
        nodes[node] = node;

        distances[node] = INF;
        prevNodes[node] = INF;
    }
    distances[0] = 0;

    auto indexOf = [&](auto nodeName) { return std::find(nodesNames.begin(), nodesNames.end(), nodeName) - nodesNames.begin(); };
    auto isVisited = [&](auto node) { return unvisited.find(node) == unvisited.end(); };
    auto isNeighbour = [&](auto currentNode, auto node) { return weights[currentNode][node] != -1; };

    auto initialNode = indexOf(initialNodeName);
    auto currentNode = initialNode;
    auto goalNode = indexOf(goalNodeName);

    while (1) {
        for (auto node = 0u; node < nodesCount; ++node) {
            if (isVisited(node)) {
                continue;
            }

            if (isNeighbour(currentNode, node)) {
                auto nodeDistance = weights[currentNode][node];
                auto totalCostToNode = distances[currentNode] + nodeDistance;

                if (totalCostToNode < distances[node]) {
                    distances[node] = totalCostToNode;
                    prevNodes[node] = currentNode;
                }
            }
        }

        if (currentNode == goalNode) {
            std::cout << "Goal node found" << std::endl;

            std::vector<int> stack;
            while (currentNode != initialNode) {
                stack.push_back(currentNode);

                auto prev = prevNodes[currentNode];
                currentNode = prev;
            }

            std::cout << "Total cost: " << distances[goalNode] << std::endl;
            std::cout << "Path: " << nodesNames[currentNode];



            for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
                auto nextNodeName = nodesNames[*it];
                std::cout << " -> " << nextNodeName;
            }
            std::cout << std::endl;

            break;
        }

        unvisited.erase(currentNode);

        auto minCost = std::numeric_limits<int>::max();
        auto nextNode = -1;

        for (auto node = 0u; node < nodesCount; ++node) {
            auto totalCost = distances[node];

            if (!isVisited(node) && totalCost < minCost) {
                minCost = totalCost;
                nextNode = node;
            }
        }

        currentNode = nextNode;

        if (currentNode == -1) {
            std::cout << "Path not found" << std::endl;
            return;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}