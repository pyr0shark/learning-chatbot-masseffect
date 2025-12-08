class Chatbot {
    constructor() {
        this.messagesContainer = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.apiEndpoint = '/api/chat'; // Backend API endpoint (same server)
        this.sessionId = this.getOrCreateSessionId(); // Session ID for conversation history
        this.displayedFactIds = new Set(); // Track which facts have been displayed
        this.factEventSource = null; // SSE connection
        
        this.initializeEventListeners();
        this.autoResizeTextarea();
        
        // Start SSE connection for real-time fact updates
        this.startFactStream();
    }
    
    getOrCreateSessionId() {
        // Generate a new session ID on each page load
        // This ensures conversation history, pending facts, and used search results are cleared on refresh
        // Only facts added to the index persist (they're saved to index.json)
        const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        return sessionId;
    }
    
    initializeEventListeners() {
        // Send button click
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Enter key to send (Shift+Enter for new line)
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => this.autoResizeTextarea());
    }
    
    autoResizeTextarea() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        
        if (!message) return;
        
        // Clear input
        this.messageInput.value = '';
        this.autoResizeTextarea();
        
        // Disable input
        this.setInputEnabled(false);
        
        // Add user message
        this.addMessage(message, 'user');
        
        // Ensure SSE connection is active (fact discovery starts on backend immediately)
        if (!this.factEventSource || this.factEventSource.readyState === EventSource.CLOSED) {
            this.startFactStream();
        }
        
        // Show typing indicator
        const typingId = this.showTypingIndicator();
        
        try {
            // Get bot response
            const responseData = await this.getBotResponse(message);
            
            // Remove typing indicator
            this.removeTypingIndicator(typingId);
            
            // Add bot response
            this.addMessage(responseData.response || responseData, 'bot');
        } catch (error) {
            // Remove typing indicator
            this.removeTypingIndicator(typingId);
            
            // Show error message
            this.addMessage(
                'Sorry, an error occurred. Please try again.',
                'bot'
            );
            console.error('Error:', error);
        } finally {
            // Re-enable input
            this.setInputEnabled(true);
            this.messageInput.focus();
        }
    }
    
    async getBotResponse(message) {
        try {
            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message: message,
                    session_id: this.sessionId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            return data; // Return full response object
        } catch (error) {
            // Log the actual error for debugging
            console.error('API call failed:', error);
            console.error('Error details:', {
                message: error.message,
                stack: error.stack,
                name: error.name
            });
            // Don't use mock response - show actual error to user
            throw error; // Re-throw so the error handler in sendMessage can show it
        }
    }
    
    getMockResponse(message) {
        // Mock responses voor testing zonder backend
        const lowerMessage = message.toLowerCase();
        let responseText = '';
        
        if (lowerMessage.includes('hallo') || lowerMessage.includes('hi')) {
            responseText = 'Hello! I\'m your Mass Effect lore assistant. I can answer questions about the Mass Effect universe. What would you like to know?';
        } else if (lowerMessage.includes('hoe gaat het')) {
            responseText = 'Het gaat goed, dank je! Hoe gaat het met jou?';
        } else if (lowerMessage.includes('help') || lowerMessage.includes('help')) {
            responseText = 'Ik ben hier om je te helpen! Stel gerust je vraag.';
        } else if (lowerMessage.includes('bedankt') || lowerMessage.includes('dank')) {
            responseText = 'Graag gedaan! Is er nog iets anders waar ik je mee kan helpen?';
        } else {
            responseText = `Je vroeg: "${message}". Dit is een mock response. Verbind de frontend met je backend API om echte antwoorden te krijgen.`;
        }
        
        return { response: responseText, pending_facts_count: 0 };
    }
    
    addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = sender === 'user' ? '👤' : '🤖';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        // Format message (support basic markdown-like formatting)
        const formattedText = this.formatMessage(text);
        content.innerHTML = formattedText;
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    formatMessage(text) {
        // Basic formatting: convert newlines to <br>, code blocks, etc.
        let formatted = text
            .replace(/\n/g, '<br>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/\*([^*]+)\*/g, '<em>$1</em>');
        
        // Wrap code blocks
        formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        return `<p>${formatted}</p>`;
    }
    
    showTypingIndicator() {
        const typingId = 'typing-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.id = typingId;
        messageDiv.className = 'message bot-message';
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = '🤖';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        return typingId;
    }
    
    removeTypingIndicator(typingId) {
        const typingElement = document.getElementById(typingId);
        if (typingElement) {
            typingElement.remove();
        }
    }
    
    setInputEnabled(enabled) {
        this.messageInput.disabled = !enabled;
        this.sendButton.disabled = !enabled;
    }
    
    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    startFactStream() {
        // Close existing connection if any
        if (this.factEventSource) {
            this.factEventSource.close();
        }
        
        // Create new SSE connection
        const url = `/api/facts/stream?session_id=${encodeURIComponent(this.sessionId)}`;
        console.log('Starting SSE connection for facts:', url);
        this.factEventSource = new EventSource(url);
        
        this.factEventSource.onmessage = (event) => {
            try {
                const factData = JSON.parse(event.data);
                console.log('Received fact via SSE:', factData.fact_id, factData.fact.substring(0, 50));
                
                // Display fact if not already shown
                if (factData.status === 'pending' && !this.displayedFactIds.has(factData.fact_id)) {
                    const existingFact = document.querySelector(`[data-fact-id="${factData.fact_id}"]`);
                    if (!existingFact) {
                        this.displayPendingFact(factData);
                        this.displayedFactIds.add(factData.fact_id);
                    }
                }
            } catch (error) {
                console.error('Error parsing SSE message:', error);
            }
        };
        
        this.factEventSource.onerror = (error) => {
            console.error('SSE connection error:', error);
            // Connection will auto-reconnect, but we can also manually reconnect if needed
            if (this.factEventSource.readyState === EventSource.CLOSED) {
                console.log('SSE connection closed, will attempt to reconnect...');
                setTimeout(() => {
                    if (this.factEventSource.readyState === EventSource.CLOSED) {
                        this.startFactStream();
                    }
                }, 3000);
            }
        };
        
        this.factEventSource.onopen = () => {
            console.log('SSE connection opened for facts');
        };
    }
    
    displayPendingFact(factData) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message fact-message';
        messageDiv.dataset.factId = factData.fact_id;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = '📝';
        
        const content = document.createElement('div');
        content.className = 'message-content fact-content';
        
        // Fact text (editable)
        const factText = document.createElement('div');
        factText.className = 'fact-text';
        factText.innerHTML = `<strong>New fact discovered:</strong><br>${this.formatMessage(factData.fact)}`;
        
        // Edit input (hidden by default)
        const editInput = document.createElement('textarea');
        editInput.className = 'fact-edit-input';
        editInput.value = factData.fact;
        editInput.style.display = 'none';
        editInput.rows = 3;
        
        // Buttons container
        const buttonsContainer = document.createElement('div');
        buttonsContainer.className = 'fact-buttons';
        
        // Edit button
        const editBtn = document.createElement('button');
        editBtn.className = 'fact-btn fact-edit-btn';
        editBtn.textContent = 'Edit';
        editBtn.onclick = () => this.toggleFactEdit(factData.fact_id);
        
        // Approve button
        const approveBtn = document.createElement('button');
        approveBtn.className = 'fact-btn fact-approve-btn';
        approveBtn.textContent = 'Approve';
        approveBtn.onclick = () => {
            const currentValue = editInput.style.display === 'none' 
                ? factData.fact 
                : editInput.value;
            this.approveFact(factData.fact_id, currentValue);
        };
        
        // Reject button
        const rejectBtn = document.createElement('button');
        rejectBtn.className = 'fact-btn fact-reject-btn';
        rejectBtn.textContent = 'Reject';
        rejectBtn.onclick = () => this.rejectFact(factData.fact_id);
        
        buttonsContainer.appendChild(editBtn);
        buttonsContainer.appendChild(approveBtn);
        buttonsContainer.appendChild(rejectBtn);
        
        content.appendChild(factText);
        content.appendChild(editInput);
        content.appendChild(buttonsContainer);
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    toggleFactEdit(factId) {
        const factMessage = document.querySelector(`[data-fact-id="${factId}"]`);
        if (!factMessage) return;
        
        const factText = factMessage.querySelector('.fact-text');
        const editInput = factMessage.querySelector('.fact-edit-input');
        const editBtn = factMessage.querySelector('.fact-edit-btn');
        
        if (editInput.style.display === 'none') {
            editInput.style.display = 'block';
            factText.style.display = 'none';
            editBtn.textContent = 'Cancel';
            editInput.focus();
        } else {
            editInput.style.display = 'none';
            factText.style.display = 'block';
            editBtn.textContent = 'Edit';
        }
    }
    
    async approveFact(factId, editedFact = null) {
        try {
            const response = await fetch('/api/facts/approve', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    fact_id: factId,
                    edited_fact: editedFact || null
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to approve fact');
            }
            
            // Get fact text before removing the message
            const factMessage = document.querySelector(`[data-fact-id="${factId}"]`);
            let factText = '';
            if (factMessage) {
                const editInput = factMessage.querySelector('.fact-edit-input');
                const factTextElement = factMessage.querySelector('.fact-text');
                // Get the current fact text (either from edit input if visible, or from fact text element)
                if (editInput && editInput.style.display !== 'none') {
                    factText = editInput.value;
                } else if (factTextElement) {
                    // Extract text from the fact text element (remove the "New fact discovered:" prefix)
                    const textContent = factTextElement.textContent || factTextElement.innerText;
                    factText = textContent.replace(/^New fact discovered:\s*/i, '').trim();
                }
                factMessage.remove();
            }
            
            // Show confirmation with fact text
            const approvalMessage = factText 
                ? `✅ Fact approved and added to index!\n\n📝 **Fact:** ${factText}`
                : '✅ Fact approved and added to index!';
            this.addMessage(approvalMessage, 'bot');
        } catch (error) {
            console.error('Error approving fact:', error);
            this.addMessage('❌ Error approving fact. Please try again.', 'bot');
        }
    }
    
    async rejectFact(factId) {
        try {
            const response = await fetch('/api/facts/reject', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    fact_id: factId
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to reject fact');
            }
            
            // Get fact text before removing the message
            const factMessage = document.querySelector(`[data-fact-id="${factId}"]`);
            let factText = '';
            if (factMessage) {
                const editInput = factMessage.querySelector('.fact-edit-input');
                const factTextElement = factMessage.querySelector('.fact-text');
                // Get the current fact text (either from edit input if visible, or from fact text element)
                if (editInput && editInput.style.display !== 'none') {
                    factText = editInput.value;
                } else if (factTextElement) {
                    // Extract text from the fact text element (remove the "New fact discovered:" prefix)
                    const textContent = factTextElement.textContent || factTextElement.innerText;
                    factText = textContent.replace(/^New fact discovered:\s*/i, '').trim();
                }
                factMessage.remove();
            }
            
            // Show rejection message with fact text
            const rejectionMessage = factText 
                ? `❌ Fact rejected.\n\n📝 **Fact:** ${factText}`
                : '❌ Fact rejected.';
            this.addMessage(rejectionMessage, 'bot');
        } catch (error) {
            console.error('Error rejecting fact:', error);
            this.addMessage('❌ Error rejecting fact. Please try again.', 'bot');
        }
    }
}

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new Chatbot();
});

