import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../config';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const Chat = ({ chatLog, codeEnhancement }) => {
    const [conversation, setConversation] = useState([]);
    const [typedMessage, setTypedMessage] = useState('');
    const [isWaitingResponse, setIsWaitingResponse] = useState(false);
    const scrollAnchor = useRef(null);

    useEffect(() => {
        if (chatLog) {
            setConversation(chatLog);
        }
    }, [chatLog]);

    useEffect(() => {
        scrollAnchor.current?.scrollIntoView({ behavior: "smooth" });
    }, [conversation]);

    const sendMessage = async () => {
        if (!typedMessage.trim() || isWaitingResponse) return;

        const newTask = { role: 'user', content: typedMessage };
        setConversation(prev => [...prev, newTask]);
        setTypedMessage('');
        setIsWaitingResponse(true);

        try {
            const apiResponse = await axios.post(`${API_BASE_URL}/api/chat`, { message: newTask.content, enhance_code: codeEnhancement });
            setConversation(apiResponse.data.history);
        } catch (err) {
            console.error("Communication failure", err);
            setConversation(prev => [...prev, { role: 'assistant', content: "Error: Unable to reach AI service." }]);
        } finally {
            setIsWaitingResponse(false);
        }
    };

    const checkKey = (e) => {
        if (e.key === 'Enter') sendMessage();
    };

    return (
        <div className="chat-container">
            <div className="chat-messages">
                {conversation.map((entry, index) => (
                    <div key={index} className={`message ${entry.role}`}>
                        <ReactMarkdown
                            children={entry.content}
                            components={{
                                code({ node, inline, className, children, ...props }) {
                                    const langMatch = /language-(\w+)/.exec(className || '')
                                    let codeString = String(children).replace(/\n$/, '');

                                    // Check if code already has line numbers (format: "  123 | code")
                                    const lineNumberRegex = /^\s*(\d+)\s*\|\s*/;
                                    const hasLineNumbers = lineNumberRegex.test(codeString);

                                    let startingLine = 1;
                                    let cleanedCode = codeString;

                                    if (hasLineNumbers) {
                                        // Extract the starting line number
                                        const firstLineMatch = codeString.match(lineNumberRegex);
                                        if (firstLineMatch) {
                                            startingLine = parseInt(firstLineMatch[1], 10);
                                        }

                                        // Remove line numbers from each line
                                        cleanedCode = codeString
                                            .split('\n')
                                            .map(line => line.replace(lineNumberRegex, ''))
                                            .join('\n');
                                    }

                                    return !inline && langMatch ? (
                                        <SyntaxHighlighter
                                            {...props}
                                            children={cleanedCode}
                                            style={vscDarkPlus}
                                            language={langMatch[1]}
                                            PreTag="div"
                                            showLineNumbers={true}
                                            startingLineNumber={startingLine}
                                            wrapLines={true}
                                            customStyle={{
                                                margin: 0,
                                                borderRadius: 0,
                                                fontSize: '13px',
                                                lineHeight: '1.5'
                                            }}
                                        />
                                    ) : (
                                        <code {...props} className={className}>
                                            {children}
                                        </code>
                                    )
                                }
                            }}
                        />
                    </div>
                ))}
                {isWaitingResponse && <div className="message assistant">Typing...</div>}
                <div ref={scrollAnchor} />
            </div>
            <div className="chat-input">
                <input
                    type="text"
                    value={typedMessage}
                    onChange={(e) => setTypedMessage(e.target.value)}
                    onKeyDown={checkKey}
                    placeholder="Ask about the code..."
                />
                <button onClick={sendMessage} disabled={isWaitingResponse}>SEND</button>
            </div>
        </div>
    );
};

export default Chat;
