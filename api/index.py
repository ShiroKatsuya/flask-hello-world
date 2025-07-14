from flask import render_template, request, jsonify, Response, stream_with_context, session, redirect, url_for
from app import app
from MultiAgent import app as multi_agent_app  # Import the Langchain graph app
from MultiAgent import model_alice, model_bob, model_supervisor, alice_simple_prompt, bob_simple_prompt, supervisor_routing_prompt, is_difficult_question, web_tools, alice_tool_executor, bob_tool_executor
from langchain.schema import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import json
import os
from datetime import datetime

from flask import Flask


app = Flask(__name__)

@app.route('/')
def index():
    """Main page route that renders the search interface"""
    return render_template('index.html')

@app.route('/chat')
def chat():
    """Route for the Chat page"""
    return render_template('index.html')


@app.route('/clear-conversation')
def clear_conversation():
    """Clear conversation history and redirect to chat page"""
    session.pop('conversation_messages', None)
    session.pop('conversation_title', None)
    return redirect(url_for('chat'))

@app.route('/discover')
def discover():
    """Route for the Discover page"""
    return render_template('discover.html')

@app.route('/spaces')
def spaces():
    """Route for the Spaces page"""
    return render_template('spaces.html')

@app.route('/account')
def account():
    """Route for the Account page"""
    return render_template('account.html')

@app.route('/upgrade')
def upgrade():
    """Route for the Upgrade page"""
    return render_template('upgrade.html')

@app.route('/install')
def install():
    """Route for the Install page"""
    return render_template('install.html')


@app.route('/action/<action_type>', methods=['POST'])
def handle_action(action_type):
    """Handle different action button clicks"""
    query = request.form.get('query', '').strip()
    
    actions = {
        'health': 'Health analysis',
        'summarize': 'Content summarization',
        'analyze': 'Data analysis',
        'plan': 'Planning assistance',
        'local': 'Local search'
    }
    
    if action_type not in actions:
        return jsonify({'error': 'Invalid action type'}), 400
    
    if not query:
        return jsonify({'error': 'Please enter a query first'}), 400
    
    return jsonify({
        'action': actions[action_type],
        'query': query,
        'message': f'{actions[action_type]} would be performed for: "{query}"'
    })

@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    """Handle chat requests and stream responses from the Multi-Agent system using the full LangGraph workflow."""
    data = request.get_json()
    query = data.get('query', '').strip()
    is_new_conversation = data.get('is_new_conversation', True)

    if not query:
        return jsonify({'error': 'Please enter a query'}), 400

    def generate():
        # Get existing conversation history from session or start fresh
        if is_new_conversation:
            # Start new conversation
            conversation_messages = [HumanMessage(content=query)]
            session['conversation_messages'] = [HumanMessage(content=query)]
            session['conversation_title'] = query
        else:
            # Continue existing conversation
            conversation_messages = session.get('conversation_messages', [])
            # Add the new query to conversation history
            conversation_messages.append(HumanMessage(content=query))
        
        current_agent = "Supervisor"
        
        # Continue conversation until supervisor decides to finish
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            if current_agent == "Supervisor":
                # Supervisor makes decision
                alice_workload = session.get('alice_workload', 0)
                bob_workload = session.get('bob_workload', 0)
                supervisor_input = {
                    "messages": conversation_messages,
                    "input": conversation_messages[-1].content,
                    "members": "Alice, Bob",
                    "alice_workload": alice_workload,
                    "bob_workload": bob_workload,
                }
                
                # Stream supervisor's decision
                full_decision = ""
                for chunk in model_supervisor.stream(supervisor_routing_prompt.format_messages(**supervisor_input)):
                    if hasattr(chunk, 'content') and chunk.content:
                        full_decision += chunk.content
                        yield f"data: {json.dumps({'sender': 'Supervisor', 'content': chunk.content, 'type': 'chunk'})}\n\n"
                
                # Send complete message for supervisor
                yield f"data: {json.dumps({'sender': 'Supervisor', 'content': full_decision, 'type': 'complete'})}\n\n"
                
                # Add supervisor message to conversation
                supervisor_message = AIMessage(content=full_decision, name="Supervisor")
                conversation_messages.append(supervisor_message)
                # Update session with conversation history
                session['conversation_messages'] = conversation_messages
                
                # Parse supervisor's decision
                def parse_supervisor_decision(decision):
                    decision_lower = decision.lower()
                    if "route_to:" in decision_lower:
                        for member in ["Alice", "Bob"]:
                            if member.lower() in decision_lower:
                                return member
                    if "finish" in decision_lower:
                        return "FINISH"
                    for member in ["Alice", "Bob"]:
                        if member.lower() in decision_lower:
                            return member
                    return "Alice"  # Default fallback

                next_agent_name = parse_supervisor_decision(full_decision)
                if next_agent_name in ["Alice", "Bob"]:
                    current_agent = next_agent_name
                    # Update workload in session
                    if current_agent == "Alice":
                        session['alice_workload'] = alice_workload + 1
                    elif current_agent == "Bob":
                        session['bob_workload'] = bob_workload + 1
                    # Send routing message
                    yield f"data: {{'sender': 'Supervisor', 'content': 'ROUTE_TO: {next_agent_name} - Routing to {next_agent_name}', 'type': 'complete'}}\n\n"
                    continue
                elif next_agent_name == "FINISH":
                    yield f"data: {{'sender': 'Supervisor', 'content': 'ROUTE_TO: FINISH - Conversation complete', 'type': 'complete'}}\n\n"
                    break
                else:
                    # Default to finish if no clear decision
                    yield f"data: {{'sender': 'Supervisor', 'content': 'ROUTE_TO: FINISH - Default finish', 'type': 'complete'}}\n\n"
                    break
            
            elif current_agent in ["Alice", "Bob"]:
                # Agent responds to the current question
                current_question = conversation_messages[-1].content
                chat_history = conversation_messages[:-1]
                
                # Determine if this is a difficult question
                use_tools = is_difficult_question(current_question)
                
                if current_agent == "Alice":
                    print(f"[DEBUG] Alice starting response for question: {current_question[:50]}...")
                    print(f"[DEBUG] Alice model: {model_alice}")
                    full_content = ""  # Initialize full_content for Alice
                    if use_tools:
                        print("[DEBUG] Alice using tools")
                        # Use tool-enabled agent
                        try:
                            alice_tool_runnable = alice_tool_executor | StrOutputParser() # Assuming alice_tool_executor can be chained with StrOutputParser for streaming
                            for chunk in alice_tool_runnable.stream({
                                "input": current_question,
                                "chat_history": chat_history
                            }):
                                if chunk:
                                    full_content += chunk
                                    yield f"data: {json.dumps({'sender': 'Alice', 'content': chunk, 'type': 'chunk'})}\n\n"
                            # After streaming all chunks, send a complete message (optional, but good for clarity)
                            # yield f"data: {json.dumps({'sender': 'Alice', 'content': full_content, 'type': 'complete'})}\n\n"
                        except Exception as e:
                            print(f"[DEBUG] Alice tool execution failed: {str(e)}")
                            # Fallback to simple agent with streaming
                            alice_simple_runnable = alice_simple_prompt | model_alice | StrOutputParser()
                            for chunk in alice_simple_runnable.stream({
                                "input": current_question,
                                "chat_history": conversation_messages,
                            }):
                                if chunk:
                                    full_content += chunk
                                    yield f"data: {json.dumps({'sender': 'Alice', 'content': chunk, 'type': 'chunk'})}\n\n"
                            # Send complete message
                            # yield f"data: {json.dumps({'sender': 'Alice', 'content': full_content, 'type': 'complete'})}\n\n"
                    else:
                        print("[DEBUG] Alice using simple streaming")
                        # Use simple agent with streaming
                        alice_simple_runnable = alice_simple_prompt | model_alice | StrOutputParser()
                        chunk_count = 0
                        for chunk in alice_simple_runnable.stream({
                            "input": current_question,
                            "chat_history": conversation_messages,
                        }):
                            if chunk:
                                chunk_count += 1
                                full_content += chunk
                                yield f"data: {json.dumps({'sender': 'Alice', 'content': chunk, 'type': 'chunk'})}\n\n"
                        print(f"[DEBUG] Alice streamed {chunk_count} chunks, total length: {len(full_content)}")
                        # Send complete message
                        # yield f"data: {json.dumps({'sender': 'Alice', 'content': full_content, 'type': 'complete'})}\n\n"
                
                elif current_agent == "Bob":
                    print(f"[DEBUG] Bob starting response for question: {current_question[:50]}...")
                    print(f"[DEBUG] Bob model: {model_bob}")
                    full_content = ""  # Initialize full_content for Bob
                    if use_tools:
                        print("[DEBUG] Bob using tools")
                        # Use tool-enabled agent with streaming
                        try:
                            bob_tool_runnable = bob_tool_executor | StrOutputParser() # Assuming bob_tool_executor can be chained with StrOutputParser for streaming
                            for chunk in bob_tool_runnable.stream({
                                "input": current_question,
                                "chat_history": chat_history
                            }):
                                if chunk:
                                    full_content += chunk
                                    yield f"data: {json.dumps({'sender': 'Bob', 'content': chunk, 'type': 'chunk'})}\n\n"
                            # After streaming all chunks, send a complete message (optional, but good for clarity)
                            # yield f"data: {json.dumps({'sender': 'Bob', 'content': full_content, 'type': 'complete'})}\n\n"
                        except Exception as e:
                            print(f"[DEBUG] Bob tool execution failed: {str(e)}")
                            # Fallback to simple agent with streaming
                            bob_simple_runnable = bob_simple_prompt | model_bob | StrOutputParser()
                            for chunk in bob_simple_runnable.stream({
                                "input": current_question,
                                "chat_history": conversation_messages,
                            }):
                                if chunk:
                                    full_content += chunk
                                    yield f"data: {json.dumps({'sender': 'Bob', 'content': chunk, 'type': 'chunk'})}\n\n"
                            # Send complete message
                            # yield f"data: {json.dumps({'sender': 'Bob', 'content': full_content, 'type': 'complete'})}\n\n"
                    else:
                        print("[DEBUG] Bob using simple streaming")
                        # Use simple agent with streaming
                        bob_simple_runnable = bob_simple_prompt | model_bob | StrOutputParser()
                        chunk_count = 0
                        for chunk in bob_simple_runnable.stream({
                            "input": current_question,
                            "chat_history": conversation_messages,
                        }):
                            if chunk:
                                chunk_count += 1
                                full_content += chunk
                                yield f"data: {json.dumps({'sender': 'Bob', 'content': chunk, 'type': 'chunk'})}\n\n"
                        print(f"[DEBUG] Bob streamed {chunk_count} chunks, total length: {len(full_content)}")
                        # Send complete message
                        # yield f"data: {json.dumps({'sender': 'Bob', 'content': full_content, 'type': 'complete'})}\n\n"
                
                # Add agent response to conversation
                agent_message = AIMessage(content=full_content, name=current_agent)
                conversation_messages.append(agent_message)
                # Update session with conversation history
                session['conversation_messages'] = conversation_messages
                
                # Route back to supervisor for next decision
                current_agent = "Supervisor"
        
        # Send end stream signal
        yield "event: end_stream\n"
        yield "data: [END]\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/save_conversation', methods=['POST'])
def save_conversation():
    data = request.get_json()
    conversation = data.get('conversation', [])

    if not conversation:
        return jsonify({'status': 'error', 'message': 'No conversation data provided.'}), 400

    # Create a directory to save conversations if it doesn't exist
    save_dir = 'conversations'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate a unique filename using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f'conversation_{timestamp}.json')

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=4)
        return jsonify({'status': 'success', 'message': 'Conversation saved.', 'file_path': file_path}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/list_conversations')
def list_conversations():
    conversations_dir = 'conversations'
    if not os.path.exists(conversations_dir):
        return jsonify([])
    conversation_files = [f for f in os.listdir(conversations_dir) if f.endswith('.json')]
    return jsonify(conversation_files)


@app.route('/view_conversation/<filename>')
def view_conversation(filename):
    conversation_path = os.path.join('conversations', filename)
    if not os.path.exists(conversation_path):
        return jsonify({'status': 'error', 'message': 'Conversation not found.'}), 404
    with open(conversation_path, 'r') as f:
        conversation_data = json.load(f)

    # Transform the conversation data into the expected format
    # Extract title from filename or use a default
    title = filename.replace('.json', '').replace('conversation_', 'Conversation ')
    
    # Extract messages from the conversation data
    messages = conversation_data if isinstance(conversation_data, list) else []
    
    # Generate related questions based on the conversation content
    related_questions = []
    for message in messages:
        if message.get('type') == 'chat' and message.get('sender') == 'user':
            content = message.get('content', '')
            if content and len(content) > 10:  # Only add substantial questions
                related_questions.append(content)
    
    # Limit to 5 related questions
    related_questions = related_questions[:5]
    
    # Create the expected conversation structure
    conversation = {
        'title': title,
        'messages': messages,
        'related_questions': related_questions
    }

    return render_template('conversations.html', conversation=conversation)


@app.route('/get_conversation_history')
def get_conversation_history():
    """Get conversation history from session"""
    conversation_messages = session.get('conversation_messages', [])
    conversation_title = session.get('conversation_title', '')
    
    # Convert LangChain messages to serializable format
    serialized_messages = []
    for msg in conversation_messages:
        if hasattr(msg, 'content') and hasattr(msg, '__class__'):
            message_data = {
                'type': msg.__class__.__name__,
                'content': msg.content
            }
            if hasattr(msg, 'name') and msg.name:
                message_data['name'] = msg.name
            serialized_messages.append(message_data)
    
    return jsonify({
        'conversation_messages': serialized_messages,
        'conversation_title': conversation_title
    })

@app.route('/grpo')
def grpo():
    """Route for the GRPO page"""
    return render_template('grpo.html')
