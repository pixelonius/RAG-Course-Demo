import {ChatOpenAI} from "@langchain/openai";
import {createReactAgent} from '@langchain/langgraph/prebuilt';
import { vectorStore, addDocumentsToVectorStore } from './embeddings.js'
import { tool } from "@langchain/core/tools";
import {z} from 'zod'
import data from './data.js'
import dotenv from 'dotenv';
import { MemorySaver } from "@langchain/langgraph";

// Load .env into process.env
dotenv.config();

const video1=data[0]
await addDocumentsToVectorStore(video1);
const video_id = "imMbPxcL8NY";

//instantiating the chat model
const model = new ChatOpenAI({ 
  model: "gpt-4o",
  apiKey: process.env.OPENAI_API_KEY
});

//retrieveal tool
const retrieveTool = tool( async ({query}, {configurable: {video_id}}) => {
console.log('Retrieving docs for the quey--------------');

const retrievedDocs = await vectorStore.similaritySearch(query, 3) //metadata for the similarity search
const serializedDocs = retrievedDocs.map ((doc)=> doc.pageContent).join('\n');

return serializedDocs;
},
{ //metadata dor the LLM
    name: 'retrieve',
    description: 'Retrieve the most relevant chunks of text from the transcript of a video',
    schema: z.object({ //zod schema for the LLM
        query: z.string()
    })
})


//adding contextual memory
const memorySaver = new MemorySaver();

//creating the ReAct Agent
const agent = createReactAgent({
    llm: model,
    tools: [ retrieveTool],
    checkpointer: memorySaver
});

//trsting the agent
const results =  await agent.invoke({
    messages:[{
        role: "user",
        content: 'Who was 200 of a seconds away?'
    }]
},
{
    configurable:{thread_id: 1, video_id} //needs to know what thread is it remembering
})

console.log(results)