import React, { useState } from 'react';
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

const Chatbot = () => {
    const [messages, setMessages] = useState([
        { sender: 'bot', text: 'Hello! How can I assist you today?' }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const sendToAPI = async (userMessage) => {
        try {
            const response = await fetch('API_ENDPOINT', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: userMessage,
                    timestamp: new Date().toISOString(),
                    sessionId: Math.random().toString(36).substring(7), 
                })
            });

            if (!response.ok) {
                throw new Error('API request failed');
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error sending message to API:', error);
            return { error: 'Failed to process your message. Please try again.' };
        }
    };

    const getMLResponse = async () => {
        try {
            const response = await fetch('ML_API_ENDPOINT', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) {
                throw new Error('ML API request failed');
            }

            const data = await response.json();
            return data.response || data.message;
        } catch (error) {
            console.error('Error getting ML response:', error);
            return 'Sorry, I encountered an error processing your request.';
        }
    };

    const sendMessage = async () => {
        if (input.trim()) {
            setIsLoading(true);
            const userMessage = input.trim();
            
            setMessages(prev => [...prev, { sender: 'user', text: userMessage }]);
            setInput('');

            try {
                const apiResponse = await sendToAPI(userMessage);
                
                if (apiResponse.error) {
                    setMessages(prev => [...prev, { sender: 'bot', text: apiResponse.error }]);
                    return;
                }

                const mlResponse = await getMLResponse();
                
                setMessages(prev => [...prev, { sender: 'bot', text: mlResponse }]);
            } catch (error) {
                setMessages(prev => [...prev, { 
                    sender: 'bot', 
                    text: 'Sorry, I encountered an error. Please try again.' 
                }]);
            } finally {
                setIsLoading(false);
            }
        }
    };

    return (
        <div>
            <Dialog>
                <DialogTrigger asChild>
                    <Button className="fixed bottom-4 right-4 rounded-full w-12 h-12">
                        💬
                    </Button>
                </DialogTrigger>

                <DialogContent className="sm:max-w-[425px]">
                    <DialogHeader>
                        <DialogTitle>Chatbot</DialogTitle>
                        <DialogDescription>Ask me anything!</DialogDescription>
                    </DialogHeader>
                    <div className="flex flex-col h-[400px]">
                        <div className="flex-1 overflow-y-auto space-y-4 p-4">
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
                                        }`}
                                    >
                                        {msg.text}
                                    </div>
                                </div>
                            ))}
                        </div>
                        <div className="border-t p-4">
                            <div className="flex gap-2">
                                <Textarea
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyPress={(e) => {
                                        if (e.key === 'Enter' && !e.shiftKey) {
                                            e.preventDefault();
                                            sendMessage();
                                        }
                                    }}
                                    placeholder="Type a message..."
                                    className="flex-1 resize-none"
                                    disabled={isLoading}
                                />
                                <Button 
                                    onClick={sendMessage} 
                                    disabled={isLoading}
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
                </DialogContent>
            </Dialog>
        </div>
    );
};

export default Chatbot;