import type { Request, Response, NextFunction } from 'express';
import { cache } from '../services/caching.service';

export function cacherFactory(duration: number | string) {
  return (req: Request, res: Response, next: NextFunction) => {
    const key = '__express__' + req.originalUrl || req.url;
    const cachedValue = cache.get(key);

    if (cachedValue) {
      res.send(cachedValue);
    } else {
      const sendResponse = res.send;
      res.send = (body) => {
        cache.set(key, body, duration);
        return sendResponse(body);
      };
      next();
    }
  };
}
