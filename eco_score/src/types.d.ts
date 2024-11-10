import type { RequestHandler, ErrorRequestHandler } from 'express';
import type { AuthObject } from '@clerk/clerk-sdk-node';

type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

type AckResponse = { status: 'Success' | 'Failure'; data?: object };
type ReqHandler<TReqBody> = RequestHandler<
  object,
  AckResponse,
  TReqBody,
  object
>;
type ErrHandler = ErrorRequestHandler<object, AckResponse, object, object>;

declare global {
  namespace Express {
    interface Request {
      auth: AuthObject;
    }
  }
}
