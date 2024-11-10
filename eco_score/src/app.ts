import express from 'express';
import 'express-async-errors';
import { createServer } from 'http';

import cors from 'cors';
import helmet from 'helmet';
import { accessLogger } from './middlewares/request-logger.middleware';
import { notFoundHandler } from './middlewares/not-found.middleware';
import { globalErrorHandler } from './middlewares/error-handler.middleware';

import { userRouter, activityRouter, deviceRouter } from './controllers/routes';

import { port, allowedOrigin } from './common/config';

const app = express();
const httpServer = createServer(app);

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(helmet());
app.use(
  cors({
    origin: allowedOrigin,
  })
);
app.use(accessLogger);

app.get('/', (_req, res) => {
  res.send('<h1>Hello, World</h1>');
});

app.use('/user', userRouter);
app.use('/activity', activityRouter);
app.use('/device', deviceRouter);

app.use(notFoundHandler);
app.use(globalErrorHandler);

httpServer.listen(port, () => {
  console.log(`Server Listening to port: ${port}...`);
});
