package com.sample.service.dummy.route;
import java.util.List;
import com.sample.interfaces.RoutingList;
import com.sample.route.RootRoutingListImpl;

public class VeryLongClassNameSampleRouteImpl extends RootRoutingListImpl implements RoutingList {
    @Override
    public List<String> getRouteList() {
        return routeList;
    }
}
