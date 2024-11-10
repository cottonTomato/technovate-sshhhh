import { ReqHandler } from '../../types';
import { IGetActivitiesDto, IGetActivityDataDto } from './activity.dto';
import { db } from '../../db/db';
import { activityTypes, userActivityScore } from '../../db/schema';
import { eq, and } from 'drizzle-orm';
import { StatusCodes } from 'http-status-codes';

type GetActivitiesHandler = ReqHandler<IGetActivitiesDto>;

export const getActivitiesHandler: GetActivitiesHandler = async function (
  _req,
  res
) {
  const activities = await db.select().from(activityTypes);

  res.status(StatusCodes.OK).json({
    status: 'Success',
    data: activities,
  });
};

type GetActivityDataHandler = ReqHandler<IGetActivityDataDto>;

export const getActivityDataHandler: GetActivityDataHandler = async function (
  req,
  res
) {
  const userId = req.auth.userId!;
  const { id } = req.body;

  const data = await db
    .select()
    .from(userActivityScore)
    .where(
      and(
        eq(userActivityScore.userId, userId),
        eq(userActivityScore.activityType, id)
      )
    );

  res.status(StatusCodes.OK).json({
    status: 'Success',
    data,
  });
};
