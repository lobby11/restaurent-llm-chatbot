import express from "express";
import dotenv from "dotenv";
import path from "path";

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";

dotenv.config();

const port = 3000;
const app = express();
app.use(express.json());

const __dirname = path.resolve();

// ✅ Initialize Gemini Model
const model = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-flash",
  maxOutputTokens: 2048,
  temperature: 0.7,
  apiKey: process.env.GOOGLE_API_KEY,
});

// ✅ Define Dynamic Tool - FIXED
const getMenuTool = new DynamicStructuredTool({
  name: "getMenuTool",
  description:
    "Returns today's menu for the given category. Use this tool when users ask for breakfast, lunch, evening snacks, dinner menu, or any food-related queries.",
  schema: z.object({
    category: z
      .string()
      .describe("Type of food category: breakfast, lunch, evening, or dinner"),
  }),
  func: async ({ category }) => {
    const menus = {
      breakfast: "Paratha, Tea, Fruits, Bread with Butter",
      lunch: "Rice, Dal, Curry, Roti, Salad",
      evening: "Samosa, Chutney, Tea, Biscuits", // Fixed mapping
      "evening snacks": "Samosa, Chutney, Tea, Biscuits", // Added alternative
      dinner: "Biryani, Raita, Papad, Salad",
    };

    const normalizedCategory = category.toLowerCase().trim();
    
    // Handle variations in input
    if (normalizedCategory.includes("evening") || normalizedCategory.includes("snack")) {
      return menus["evening"];
    }
    
    // FIXED: Added return statement
    return (
      menus[normalizedCategory] ||
      "Sorry, we couldn't find the menu for that category. Available categories: breakfast, lunch, evening snacks, dinner"
    );
  },
});

// ✅ Create Enhanced Prompt Template - IMPROVED
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system", 
    "You are a helpful assistant that uses tools when needed."
  ],
  ["human", "{input}"],
  ["placeholder", "{agent_scratchpad}"], // Changed from "ai" to "placeholder"
]);

// ✅ Create Agent with Tools
const agent = await createToolCallingAgent({
  llm: model,
  tools: [getMenuTool],
  prompt,
});

// ✅ Create Agent Executor
const executor = await AgentExecutor.fromAgentAndTools({
  agent,
  tools: [getMenuTool],
  verbose: true,
  maxIterations: 5, // Increased from 3
  returnIntermediateSteps: true, // Fixed property name
}); 

// ✅ Serve Frontend (index.html)
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// ✅ Chat API Endpoint - IMPROVED ERROR HANDLING
app.post("/api/chat", async (req, res) => {
  const userInput = req.body.input;
  console.log("User input:", userInput);

  if (!userInput || userInput.trim() === "") {
    return res.status(400).json({
      output: "Please provide a valid input."
    });
  }

  try {
    const response = await executor.invoke({ input: userInput });
    console.log("Agent full response:", response);

    // Primary response - agent completed successfully
    if (response.output && !response.output.includes('agent stopped due to max iterations')) {
      return res.json({ output: response.output });
    }

    // Fallback - try to get data from intermediate steps
    if (response.intermediateSteps && response.intermediateSteps.length > 0) {
      const lastStep = response.intermediateSteps[response.intermediateSteps.length - 1];
      if (lastStep && lastStep.observation) {
        return res.json({ output: lastStep.observation });
      }
    }

    // Final fallback
    return res.status(500).json({
      output: "I couldn't process your request. Please try asking for a specific menu like 'breakfast menu' or 'evening snacks menu'."
    });

  } catch (err) {
    console.error("Error during agent execution:", err);
    res.status(500).json({
      output: "Sorry, something went wrong. Please try again with a specific menu request."
    });
  }
});

// ✅ Start Express Server
app.listen(port, () => {
  console.log(`✅ Server is running on http://localhost:${port}`);
});