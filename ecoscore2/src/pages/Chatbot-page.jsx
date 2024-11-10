import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger
} from "@/components/ui/dialog";
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Loader2 } from 'lucide-react';
import { useUser } from "@clerk/clerk-react";

const chatEvents = {
    listeners: new Set(),
    subscribe(listener) {
        this.listeners.add(listener);
        return () => this.listeners.delete(listener);
    },
    emit() {
        this.listeners.forEach(listener => listener());
    }
};

export const openChat = () => {
    chatEvents.emit();
};

const ChatbotPage = () => {
    const navigate = useNavigate();
    const { user, isLoaded } = useUser();
    let userName = isLoaded ? user.fullName : "User";
    const userFirstName = isLoaded ? user.fullName.split(' ')[0] : "User";
    userName = "Mohan Kumar";

    const [messages, setMessages] = useState([
        { sender: 'bot', text: `Hello, ${userFirstName}! How can I assist you today?` }
]);

    
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isOpen, setIsOpen] = useState(false);

    useEffect(() => {
        const unsubscribe = chatEvents.subscribe(() => {
            setIsOpen(true);
        });
        return () => unsubscribe();
    }, []);

    const sendToAPI = async (userMessage) => {
        const response = await fetch('http://127.0.0.1:5000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_name: userName,
                query: userMessage,
                timestamp: new Date().toISOString(),
                sessionId: Math.random().toString(36).substring(7),
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    };

    const sendMessage = async () => {
        if (!input.trim()) return;

        try {
            setIsLoading(true);
            const userMessage = input.trim();
            
            // Add user message to chat
            setMessages(prev => [...prev, { sender: 'user', text: userMessage }]);
            setInput('');

            // Send message to API and get response
            const apiResponse = await sendToAPI(userMessage);
            
            // First add the message if it exists
            if (apiResponse.message) {
                setMessages(prev => [...prev, { 
                    sender: 'bot', 
                    text: apiResponse.message
                }]);
            }
            
            // Then add the reply if it exists
            if (apiResponse.reply) {
                setMessages(prev => [...prev, { 
                    sender: 'bot', 
                    text: apiResponse.reply
                }]);
            }

        } catch (error) {
            console.error('Error in chat:', error);
            setMessages(prev => [...prev, { 
                sender: 'bot', 
                text: 'Sorry, I encountered an error. Please try again.' 
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    useEffect(() => {
        const messageContainer = document.querySelector('.message-container');
        if (messageContainer) {
            messageContainer.scrollTop = messageContainer.scrollHeight;
        }
    }, [messages]);

    const formatBotMessage = (text) => {
        // Convert markdown-style formatting to JSX
        return text.split('\n').map((line, index) => (
            <React.Fragment key={index}>
                {line}
                <br />
            </React.Fragment>
        ));
    };

    return (
        <div className="flex items-center justify-center min-h-screen">
            <Dialog open={isOpen} onOpenChange={setIsOpen}>
                <DialogTrigger asChild>
                    <Button className="bg-green-600 hover:bg-green-700">
                        Eco is Here To Help!
                    </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-[500px]">
                    <DialogHeader>
                        <DialogTitle>Chat Assistant</DialogTitle>
                        <DialogDescription>
                            Ask me anything and I'll help you out!
                        </DialogDescription>
                    </DialogHeader>
                    <div className="flex flex-col h-[400px]">
                        <div className="message-container flex-1 overflow-y-auto space-y-4 p-4">
                            {messages.map((msg, index) => (
                                <div
                                    key={index}
                                    className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                                >
                                    <div
                                        className={`max-w-[80%] rounded-lg p-3 ${
                                            msg.sender === 'user' 
                                                ? 'bg-blue-500 text-white' 
                                                : 'bg-gray-100 text-gray-900'
                                        } whitespace-pre-wrap`}
                                    >
                                        {msg.sender === 'bot' ? formatBotMessage(msg.text) : msg.text}
                                    </div>
                                </div>
                            ))}
                        </div>
                        <div className="border-t p-4">
                            <div className="flex gap-2">
                                <Textarea
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyPress={handleKeyPress}
                                    placeholder="Type a message..."
                                    className="flex-1 resize-none"
                                    disabled={isLoading}
                                    rows={3}
                                />
                                <Button 
                                    onClick={sendMessage} 
                                    disabled={isLoading || !input.trim()}
                                    className="self-end"
                                >
                                    {isLoading ? (
                                        <Loader2 className="h-4 w-4 animate-spin" />
                                    ) : (
                                        'Send'
                                    )}
                                </Button>
                            </div>
                        </div>
                    </div>
                    <div className="mt-4">
                        <Button 
                            onClick={() => navigate('/dashboard')}
                            className="w-full bg-gray-600 hover:bg-gray-700"
                        >
                            Go Back to Home
                        </Button>
                    </div>
                </DialogContent>
            </Dialog>
        </div>
    );
};

export default ChatbotPage;